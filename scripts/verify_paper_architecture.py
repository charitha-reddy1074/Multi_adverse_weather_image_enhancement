"""Trace the current reconstruction and report known primary-paper mismatches.

This is a reproducible diagnostic, not a certification that the paper is matched.
Figure topology is reviewed visually; hooks establish what the code executes.
"""
import argparse
import ast
import hashlib
import inspect
import json
import sys
from pathlib import Path
from unittest.mock import patch

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.airformer import AIRFormer
from models.attention import QueriedAttention
from losses.multidomain import MultiDomainConsistencyLoss


class StripDocstrings(ast.NodeTransformer):
    def generic_visit(self, node):
        node = super().generic_visit(node)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body = node.body[1:]
        return node


def semantic_fingerprints():
    return {path.as_posix(): hashlib.sha256(ast.dump(StripDocstrings().visit(ast.parse(
        path.read_text(encoding="utf8")))).encode()).hexdigest()
        for directory in ("models", "losses") for path in Path(directory).glob("*.py")}


def attention_trace(mode, token_cap):
    torch.manual_seed(42)
    module = QueriedAttention(8, mode, heads=2, max_spatial_tokens=token_cap)
    x = torch.randn(1, 8, 8, 12, requires_grad=True)
    projections = {}
    handles = []
    for name in ("q_proj", "k_proj", "v_proj"):
        def capture(_, __, result, name=name):
            projections.setdefault(name, []).append(list(result.shape))
        handles.append(getattr(module, name).register_forward_hook(capture))
    softmax = torch.softmax
    attention = []

    def observe(logits, dim, *args, **kwargs):
        probabilities = softmax(logits, dim, *args, **kwargs)
        attention.append(dict(logits_shape=list(logits.shape), softmax_dim=dim,
            max_row_sum_error=(probabilities.sum(dim)-1).abs().max().item()))
        return probabilities

    with patch("torch.softmax", observe):
        output = module(x)
        output.square().mean().backward()
    for handle in handles:
        handle.remove()
    return dict(mode=mode, token_cap=token_cap, input_shape=list(x.shape),
        projected_shapes=projections, attention=attention,
        query_parameter_shape=list(module.task_query.shape),
        query_gradient_l1=module.task_query.grad.abs().sum().item(),
        query_gradient_finite=bool(torch.isfinite(module.task_query.grad).all()),
        output_shape=list(output.shape))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--output", type=Path, default=Path("docs/architecture/verification.json"))
    parser.add_argument("--baseline-ast", type=Path,
                        default=Path("docs/architecture/semantic_baseline.json"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf8"))
    keys = inspect.signature(AIRFormer).parameters
    model = AIRFormer(**{key: value for key, value in cfg.items() if key in keys})
    shape_trace = []
    prior_inputs = []
    handles = []
    for name, module in model.named_modules():
        if name in {"patch_embed", "reconstruct"} or (
            name.count(".") == 2 and name.startswith(("encoder.stages.", "encoder.downsamples.", "decoder.stages."))):
            def capture(_, __, result, name=name):
                shape_trace.append(dict(module=name, output_shape=list(result.shape)))
            handles.append(module.register_forward_hook(capture))
    for index, prior in enumerate(model.supplementary_priors):
        def capture_prior(_, inputs, index=index):
            prior_inputs.append(dict(stage=index, input_shape=list(inputs[0].shape),
                input_is_zero=bool(torch.count_nonzero(inputs[0]) == 0),
                previous_shape=list(inputs[1].shape)))
        handles.append(prior.register_forward_pre_hook(capture_prior))
    sample = torch.rand(1, 3, 32, 32)
    result = model(sample)
    result.square().mean().backward()
    for handle in handles:
        handle.remove()
    current = semantic_fingerprints()
    before = json.loads(args.baseline_ast.read_text()) if args.baseline_ast.is_file() else None
    # Eq. 12 as printed: apply piecewise penalty after spatial-mean residual.
    target = torch.zeros(1, 3, 2, 2)
    prediction = torch.tensor([1., -1., 1., -1.]).reshape(1, 1, 2, 2).repeat(1, 3, 1, 1)
    actual_spatial = MultiDomainConsistencyLoss(lambda_freq=0)(prediction, target)[0].item()
    mean_residual = (prediction-target).mean(dim=(-2, -1))
    literal_spatial = torch.nn.functional.smooth_l1_loss(mean_residual, torch.zeros_like(mean_residual)).item()
    report = dict(status="NOT EXACT - DOCUMENTED DEVIATIONS", config=args.config,
        python_version=sys.version.split()[0], torch_version=torch.__version__,
        source_pdf_sha256=json.loads(Path("docs/architecture/figures/provenance.json").read_text())["source_sha256"],
        paper_depicted_encoder_block_positions=4,
        paper_feature_labels="x1,x2,x3 are encoder intermediates; x4 is the decoder output",
        implemented_encoder_stages=len(model.encoder.stages),
        implemented_decoder_stages=len(model.decoder.stages),
        parameters=sum(parameter.numel() for parameter in model.parameters()),
        paper_table13_parameters_million=58.34,
        full_model_shape_trace=shape_trace, supplementary_prior_inputs=prior_inputs,
        all_observed_parameter_gradients_finite=all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None),
        attention_traces=[attention_trace(mode, cap) for mode in ("channel", "spatial", "wavelet") for cap in (16, 10000)],
        equation12_counterexample=dict(residual=[1, -1, 1, -1],
            code_mean_elementwise_smooth_l1=actual_spatial, printed_mean_then_smooth_l1=literal_spatial),
        model_and_loss_semantic_ast_unchanged=(current == before) if before else None,
        semantic_ast_fingerprints=current,
        limitation="Finite gradients and passing tests establish execution, not paper equivalence.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps({key: report[key] for key in ("status", "parameters", "implemented_encoder_stages",
          "implemented_decoder_stages", "equation12_counterexample", "model_and_loss_semantic_ast_unchanged")}, indent=2))


if __name__ == "__main__":
    main()
