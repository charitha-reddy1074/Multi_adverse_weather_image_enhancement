"""Short synthetic GPU throughput comparison; never writes trained weights."""
import json
from pathlib import Path
import sys
import time

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from airformer import AIRFormer, MultiDomainConsistencyLoss


def main():
    cfg = yaml.safe_load(Path('configs/four_weather.yaml').read_text())
    keys = ('base_channels', 'stage_blocks', 'attention_heads', 'max_spatial_tokens',
            'query_combine', 'use_supplementary_prior', 'use_task_queries',
            'use_frequency_decoder', 'decoder_mode', 'encoder_token_mixer',
            'residual_image', 'gradient_checkpointing', 'spatial_kv_pooling')
    results = []
    for batch, accum in ((2, 4), (4, 2), (8, 1)):
        torch.manual_seed(42)
        model = AIRFormer(**{k: cfg[k] for k in keys}).cuda().train()
        opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'])
        scaler = torch.amp.GradScaler('cuda')
        loss_fn = MultiDomainConsistencyLoss(cfg['lambda_freq'])
        x = torch.rand(batch, 3, cfg['crop_size'], cfg['crop_size'], device='cuda')
        target = x * .8
        torch.cuda.reset_peak_memory_stats()
        def update():
            opt.zero_grad(set_to_none=True)
            for _ in range(accum):
                with torch.autocast('cuda'):
                    loss = loss_fn(model(x), target)[0] / accum
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite benchmark loss')
                scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scaler.step(opt)
            scaler.update()
        for _ in range(2):
            update()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(8):
            update()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        result = dict(batch_size=batch, accumulation=accum,
                      images_per_second=64 / elapsed,
                      peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                      training_hours_20_epochs=58766 * 20 * elapsed / 64 / 3600)
        print(json.dumps(result), flush=True)
        results.append(result)
        del model, opt, scaler, x, target
        torch.cuda.empty_cache()
    Path('outputs/training_batch_benchmark.json').write_text(json.dumps({
        'scope': 'Synthetic GPU-only measurement; excludes data loading, testing and startup. Concurrent startup audit may affect timings.',
        'results': results}, indent=2))


if __name__ == '__main__':
    main()
