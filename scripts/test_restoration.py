"""Durable full-resolution restoration evaluation and an automatic visual report.

Auxiliary masks are a diagnostic, not a weather classifier: ground truth marks
pixels whose input/target mean absolute RGB error exceeds a fixed threshold;
the predicted mask marks pixels changed by the model by more than that threshold.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime
import hashlib
import html
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from airformer.data import PairedWeatherDataset
from datasets.pairing import validate_pair
from losses.multidomain import MultiDomainConsistencyLoss
from utils.inference import _restore_tensor, load_model
from utils.metrics import image_metrics

THRESHOLDS = (0.05, 0.10, 0.20)
METRICS = ("psnr", "ssim", "mae", "mse", "rmse")


def markdown_report_path(output):
    output = output.resolve()
    if output.is_relative_to(ROOT / "outputs"):
        return ROOT / "docs" / "reports" / output.relative_to(ROOT / "outputs") / "report.md"
    if output.is_relative_to(ROOT):
        return ROOT / "docs" / "reports" / output.relative_to(ROOT) / "report.md"
    return output / "report.md"


def stamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_json(path, value):
    # JSON null represents an undefined metric; do not emit invalid Infinity/NaN.
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, (float, np.floating)) and not np.isfinite(x):
            return None
        return x
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(value), indent=2, allow_nan=False), encoding="utf8")
    temporary.replace(path)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def mask_counts(degraded, restored, target, threshold):
    truth = np.abs(degraded - target).mean(axis=2) > threshold
    predicted = np.abs(degraded - restored).mean(axis=2) > threshold
    return dict(tn=int(np.count_nonzero(~truth & ~predicted)),
                fp=int(np.count_nonzero(~truth & predicted)),
                fn=int(np.count_nonzero(truth & ~predicted)),
                tp=int(np.count_nonzero(truth & predicted)))


def mask_scores(counts):
    tn, fp, fn, tp = (counts[k] for k in ("tn", "fp", "fn", "tp"))
    ratio = lambda n, d: n / d if d else None
    return dict(**counts, accuracy=ratio(tp + tn, tp + tn + fp + fn),
                precision=ratio(tp, tp + fp), recall=ratio(tp, tp + fn),
                f1=ratio(2 * tp, 2 * tp + fp + fn),
                confusion_matrix=[[tn, fp], [fn, tp]])


def pair_arrays(record):
    with Image.open(record[0]) as image:
        degraded = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
    with Image.open(record[1]) as image:
        target = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
    if degraded.shape != target.shape:
        raise ValueError(f"Pair shape mismatch: {record[0]}")
    return degraded, target


def predict(model, degraded, device, pad_multiple):
    x = torch.from_numpy(degraded).permute(2, 0, 1).unsqueeze(0).to(device)
    output, elapsed = _restore_tensor(model, x, device, pad_multiple)
    if output.shape != x.shape or not torch.isfinite(output).all():
        raise RuntimeError("Model returned an invalid shape or nonfinite pixels")
    return output, output[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy(), elapsed


def preflight(dataset, train_manifest, model, device, pad_multiple, output, signature):
    path = output / "preflight.json"
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf8"))
        if cached.get("signature") == signature and cached.get("passed"):
            print("Reusing completed preflight for the same checkpoint, manifests, and code", flush=True)
            return cached
    # Reject shared scenes or paths before evaluating. Pixel-level split checks
    # already ran in training; this check revalidates manifest identities.
    train = PairedWeatherDataset(".", None, False, manifest=train_manifest)
    train_paths = {str(p.resolve()).casefold() for r in train.records for p in r[:2]}
    train_scenes = {(r[3].casefold(), r[5]) for r in train.records}
    seen = set()
    for record in dataset.records:
        if record[4] != "test":
            raise ValueError(f"Non-test record: {record}")
        identity = str(record[0].resolve()).casefold()
        if identity in seen:
            raise ValueError(f"Duplicate test input: {record[0]}")
        seen.add(identity)
        if any(str(p.resolve()).casefold() in train_paths for p in record[:2]):
            raise ValueError(f"Train/test path overlap: {record[0]}")
        if (record[3].casefold(), record[5]) in train_scenes:
            raise ValueError(f"Train/test scene overlap: {record[0]}")
    largest = {}
    started = time.perf_counter()
    def check(item):
        i, record = item
        valid, size, other = validate_pair(record[0], record[1])
        if not valid:
            raise ValueError(f"Unreadable or mismatched pair {record[0]}: {size}, {other}")
        if min(size) < 3:
            raise ValueError(f"Image too small for SSIM: {record[0]}")
        return i, record[2], size
    with ThreadPoolExecutor(max_workers=4) as pool:
        for done, (i, weather, size) in enumerate(pool.map(check, enumerate(dataset.records)), 1):
            # Include padding when selecting the worst memory shape.
            area = ((size[0] + pad_multiple - 1) // pad_multiple) * ((size[1] + pad_multiple - 1) // pad_multiple)
            if weather not in largest or area > largest[weather][0]:
                largest[weather] = (area, i, size)
            if done % 500 == 0:
                print(f"Preflight decoded {done}/{len(dataset)} pairs", flush=True)
                write_json(output / "status.json", dict(state="preflight", pid=os.getpid(),
                    checked_pairs=done, total=len(dataset), updated_at=stamp()))
    smoke = []
    for weather, (_, index, size) in sorted(largest.items()):
        degraded, target = pair_arrays(dataset.records[index])
        prediction, restored, seconds = predict(model, degraded, device, pad_multiple)
        psnr, ssim = image_metrics(restored, target)
        with torch.inference_mode():
            loss = MultiDomainConsistencyLoss()(prediction,
                torch.from_numpy(target).permute(2, 0, 1).unsqueeze(0).to(device))[0].item()
        if not np.isfinite([loss, psnr, ssim]).all():
            raise RuntimeError(f"Nonfinite smoke metrics for {weather}")
        smoke.append(dict(weather=weather, index=index, size=size, seconds=seconds,
                          psnr=psnr, ssim=ssim, loss=loss))
        print(f"Largest {weather} pair {size}: inference, metrics, and loss passed", flush=True)
    result = dict(passed=True, signature=signature, pairs=len(dataset),
        weather_counts=dict(Counter(r[2] for r in dataset.records)),
        split_check="scene and path identities; training previously checked clean-pixel hashes",
        decode_check="all input and target files fully decoded; matching sizes",
        largest_shape_smoke=smoke, seconds=time.perf_counter()-started, finished_at=stamp())
    write_json(path, result)
    return result


def summarize(rows):
    groups = defaultdict(list)
    groups["overall"] = rows
    for row in rows:
        groups[row["weather"]].append(row)
    result = {}
    for name, values in groups.items():
        summary = dict(images=len(values))
        for prefix in ("input_", "restored_"):
            for metric in METRICS:
                summary[prefix + metric] = float(np.mean([float(v[prefix + metric]) for v in values]))
        summary["mean_psnr_gain_db"] = float(np.mean([float(v["psnr_gain_db"]) for v in values]))
        summary["mean_ssim_gain"] = float(np.mean([float(v["ssim_gain"]) for v in values]))
        summary["psnr_improved_fraction"] = float(np.mean([float(v["psnr_gain_db"]) > 0 for v in values]))
        summary["mean_loss"] = float(np.mean([float(v["loss"]) for v in values]))
        summary["mean_inference_seconds"] = float(np.mean([float(v["inference_seconds"]) for v in values]))
        summary["auxiliary_masks"] = {}
        for threshold in THRESHOLDS:
            tag = f"{threshold:.2f}"
            counts = {key: sum(int(v[f"mask_{tag}_{key}"]) for v in values) for key in ("tn", "fp", "fn", "tp")}
            summary["auxiliary_masks"][tag] = mask_scores(counts)
        result[name] = summary
    return result


def make_report(output, rows, summary, provenance, model, dataset, device, pad_multiple):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 170, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    visuals = output / "visualizations"
    visuals.mkdir(exist_ok=True)
    def save(fig, name):
        fig.savefig(visuals / (name + ".png"), bbox_inches="tight")
        fig.savefig(visuals / (name + ".svg"), bbox_inches="tight")
        plt.close(fig)
    weather = sorted(set(r["weather"] for r in rows))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    pos = np.arange(len(weather))
    for ax, metric, label in zip(axes, ("psnr", "ssim"), ("Mean PSNR (dB)", "Mean SSIM")):
        ax.bar(pos-.18, [summary[w]["input_"+metric] for w in weather], .36, label="Degraded input", color="#96a8bb")
        ax.bar(pos+.18, [summary[w]["restored_"+metric] for w in weather], .36, label="Restored output", color="#207a89")
        ax.set_xticks(pos, [f"{w}\n(n={summary[w]['images']:,})" for w in weather])
        ax.set_ylabel(label); ax.legend(); ax.grid(axis="y", alpha=.2)
    fig.suptitle("Full test set: restoration quality compared with the degraded input")
    fig.tight_layout(); save(fig, "quality_by_weather")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    gains = np.array([float(r["psnr_gain_db"]) for r in rows])
    axes[0].hist(gains[np.isfinite(gains)], bins=60, color="#207a89")
    axes[0].axvline(0, color="#c34c42", linestyle="--")
    axes[0].set(xlabel="PSNR gain (dB)", ylabel="Test images", title="Positive values indicate improvement")
    for w in weather:
        selected = [r for r in rows if r["weather"] == w]
        axes[1].scatter([float(r["input_psnr"]) for r in selected],
            [float(r["restored_psnr"]) for r in selected], s=4, alpha=.3, label=w, rasterized=True)
    lim = axes[1].get_xlim(); axes[1].plot(lim, lim, "--", color="#c34c42")
    axes[1].set(xlabel="Input PSNR (dB)", ylabel="Restored PSNR (dB)", title="Above the diagonal is better")
    axes[1].legend(markerscale=3); fig.tight_layout(); save(fig, "improvement_distribution")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    counts = np.array(summary["overall"]["auxiliary_masks"]["0.10"]["confusion_matrix"], dtype=np.int64)
    for ax, normalized in zip(axes, (False, True)):
        data = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1) if normalized else counts
        ax.imshow(data, cmap="Blues", vmin=0, vmax=1 if normalized else None)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{data[i,j]:.1%}" if normalized else f"{counts[i,j]:,}",
                        ha="center", va="center", color="white" if data[i,j] > data.max()*.55 else "black")
        ax.set_xticks([0,1], ["No large correction", "Large correction"])
        ax.set_yticks([0,1], ["Low input error", "High input error"])
        ax.set(xlabel="Predicted correction mask", ylabel="True input-error mask",
               title="Row-normalized" if normalized else "Pixel counts")
    fig.suptitle("Auxiliary mask confusion matrix | threshold = 0.10 | not weather classification")
    fig.tight_layout(); save(fig, "auxiliary_confusion_matrix")
    galleries = []
    for w in weather:
        ordered = sorted([r for r in rows if r["weather"] == w], key=lambda r: float(r["psnr_gain_db"]))
        selections = [("Worst gain", ordered[0]), ("Median gain", ordered[len(ordered)//2]), ("Best gain", ordered[-1])]
        fig, axes = plt.subplots(3, 4, figsize=(14, 9), squeeze=False)
        for row_index, (label, row) in enumerate(selections):
            degraded, target = pair_arrays(dataset.records[int(row["index"])])
            _, restored, _ = predict(model, degraded, device, pad_multiple)
            error = np.abs(restored-target).mean(axis=2)
            for col, (image, title) in enumerate(zip((degraded, restored, target, error),
                    ("Degraded input", "Restored output", "Clean target", "Output error (0–0.25)"))):
                ax = axes[row_index, col]
                ax.imshow(image, **({"cmap":"magma", "vmin":0, "vmax":.25} if col==3 else {}))
                ax.axis("off"); ax.set_title(title)
            axes[row_index, 0].text(0, -.08, f"{label}: {float(row['psnr_gain_db']):+.2f} dB; index {row['index']}",
                                       transform=axes[row_index,0].transAxes)
        fig.suptitle(f"{w.title()} | worst, median, and best PSNR gains (no cherry-picking)")
        fig.tight_layout(); save(fig, "examples_"+w); galleries.append("examples_"+w)
    overall = summary["overall"]
    lines = ["# Epoch-20 restoration test results", "", f"Completed: {stamp()}", "",
        f"Evaluated **{len(rows):,} paired test images** at their original resolution using FP32 inference.",
        "This is the custom four-weather split; haze uses the pinned unofficial ITS mirror and is not an official SOTS benchmark.", "",
        "| Group | Images | Input PSNR | Restored PSNR | PSNR gain | Input SSIM | Restored SSIM | MAE | RMSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ["overall"] + weather:
        s = summary[name]
        lines.append(f"| {name} | {s['images']} | {s['input_psnr']:.3f} | {s['restored_psnr']:.3f} | {s['mean_psnr_gain_db']:+.3f} | {s['input_ssim']:.4f} | {s['restored_ssim']:.4f} | {s['restored_mae']:.5f} | {s['restored_rmse']:.5f} |")
    lines += ["", "PSNR and SSIM are computed on clipped float RGB [0,1], full image, no border crop. SSIM uses the existing skimage RGB protocol (7-pixel window where possible).",
        "Each image has equal weight in restoration means; weather macro means are not substituted for image means. RMSE is averaged per image. Timing excludes disk loading, CPU metrics, and report creation.",
        "", "## Auxiliary pixel-error/correction masks", "",
        "These are not weather classification scores. True positive class: mean absolute RGB(input − clean target) > threshold. Predicted positive class: mean absolute RGB(input − restored output) > threshold. Thus the scores describe where the model applied corrections to corrupted pixels, not whether restoration was successful. PSNR/SSIM measure restoration quality.",
        "Scores pool pixel counts across images; large images weigh more. Confusion matrix rows are true [low input error, high input error], columns predicted [no large correction, large correction]. Zero denominators are reported as undefined.", "",
        "| Fixed threshold | Accuracy | Precision | Recall | F1 |", "|---|---:|---:|---:|---:|"]
    for tag, s in overall["auxiliary_masks"].items():
        fmt = lambda v: f"{v:.4f}" if v is not None else "undefined"
        lines.append(f"| {tag} | {fmt(s['accuracy'])} | {fmt(s['precision'])} | {fmt(s['recall'])} | {fmt(s['f1'])} |")
    lines += ["", "Threshold 0.10 is the primary diagnostic; 0.05 and 0.20 are fixed sensitivity checks, not tuned on the test set.", "",
              "The trained model and training schedule were fixed before testing. No model selection uses test scores.", ""]
    markdown_path = markdown_report_path(output)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_visuals = Path(os.path.relpath(visuals, markdown_path.parent)).as_posix()
    for name in ["quality_by_weather", "improvement_distribution", "auxiliary_confusion_matrix"] + galleries:
        lines += [f"![{name.replace('_',' ')}]({markdown_visuals}/{name}.png)", ""]
    markdown_path.write_text("\n".join(lines), encoding="utf8")
    # A portable report, with linked PNG/SVG plots and the exact protocol.
    body = "".join(f'<figure><img src="visualizations/{name}.png"><figcaption>{html.escape(name.replace("_"," "))} · <a href="visualizations/{name}.svg">SVG</a></figcaption></figure>' for name in ["quality_by_weather", "improvement_distribution", "auxiliary_confusion_matrix"] + galleries)
    markdown_link = html.escape(Path(os.path.relpath(markdown_path, output)).as_posix(), quote=True)
    document = '<!doctype html><meta charset="utf-8"><title>Epoch-20 test results</title><style>body{max-width:1200px;margin:40px auto;padding:0 24px;font:16px/1.55 system-ui;color:#172b3d}img{width:100%}pre{white-space:pre-wrap;background:#f2f5f8;padding:20px}figure{margin:35px 0}a{color:#207a89}</style><h1>Epoch-20 restoration test results</h1><p><a href="summary.json">All metrics and provenance</a> · <a href="per_image_metrics.csv">Per-image results</a> · <a href="'+markdown_link+'">Detailed report</a></p><pre>'+html.escape("\n".join(lines).split("![")[0])+"</pre>"+body
    (output / "report.html").write_text(document, encoding="utf8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=True)
    lock = output / "evaluation.lock"
    with lock.open("x", encoding="utf8") as stream:
        stream.write(str(os.getpid()))
    status = dict(state="starting", pid=os.getpid(), started_at=stamp(), checkpoint=str(Path(args.checkpoint).resolve()))
    try:
        torch.set_num_threads(2)
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        cfg = saved["config"]
        if int(saved["epoch"]) != int(cfg["epochs"]):
            raise ValueError("Expected the fixed final-epoch checkpoint")
        model, device = load_model(args.checkpoint)
        dataset = PairedWeatherDataset(".", None, False, manifest=cfg["test_manifest"])
        pad_multiple = int(cfg.get("test_pad_multiple", 128))
        signature = {"checkpoint":digest(args.checkpoint), "test_manifest":digest(cfg["test_manifest"]),
                     "train_manifest":digest(cfg["train_manifest"]), "evaluator":digest(__file__),
                     "attention":digest(ROOT/"models/attention.py"), "inference":digest(ROOT/"utils/inference.py")}
        old_provenance = output/"provenance.json"
        if (output/"per_image_metrics.csv").exists():
            if not old_provenance.exists() or json.loads(old_provenance.read_text(encoding="utf8")).get("signature") != signature:
                raise ValueError("Existing metrics belong to different code/data/checkpoint; use a new output directory")
        # Check plotting dependencies before spending time on the full test set.
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        fig = plt.figure(); plt.close(fig)
        provenance = dict(checkpoint=status["checkpoint"], epoch=saved["epoch"], signature=signature,
            model_config=saved["model_config"], test_manifest=cfg["test_manifest"],
            torch_version=torch.__version__, device=str(device),
            gpu=torch.cuda.get_device_name(device) if device.type=="cuda" else None,
            pad_multiple=pad_multiple, precision="FP32; no autocast", tile_size=0,
            thresholds=THRESHOLDS, restoration_aggregation="equal weight per image",
            mask_aggregation="pooled pixel counts")
        write_json(output/"provenance.json", provenance)
        checked = preflight(dataset, cfg["train_manifest"], model, device, pad_multiple, output, signature)
        status.update(state="preflight_passed", total=len(dataset), weather_counts=checked["weather_counts"])
        write_json(output/"status.json", status)
        if args.preflight_only:
            print("PREFLIGHT PASSED", flush=True)
            return
        csv_path = output/"per_image_metrics.csv"
        rows = []
        if csv_path.exists():
            if not args.resume:
                raise FileExistsError("Metrics already exist; use --resume")
            with csv_path.open(newline="", encoding="utf8") as stream:
                rows = list(csv.DictReader(stream))
            for i, row in enumerate(rows):
                if int(row["index"]) != i or row["input_path"] != str(dataset.records[i][0]):
                    raise ValueError("Saved metrics do not match test manifest order")
        keys = ["index", "weather", "source_dataset", "scene_id", "input_path", "target_path", "width", "height", "loss", "inference_seconds"]
        keys += [prefix+metric for prefix in ("input_", "restored_") for metric in METRICS]
        keys += ["psnr_gain_db", "ssim_gain"]
        keys += [f"mask_{threshold:.2f}_{key}" for threshold in THRESHOLDS for key in ("tn", "fp", "fn", "tp")]
        start_index = len(rows); begin = time.perf_counter()
        loss_fn = MultiDomainConsistencyLoss(float(cfg.get("lambda_freq", .04)))
        with csv_path.open("a", newline="", encoding="utf8") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            if start_index == 0:
                writer.writeheader()
            for i in range(start_index, len(dataset)):
                record = dataset.records[i]
                degraded, target = pair_arrays(record)
                prediction, restored, seconds = predict(model, degraded, device, pad_multiple)
                with torch.inference_mode():
                    loss = loss_fn(prediction, torch.from_numpy(target).permute(2,0,1).unsqueeze(0).to(device))[0].item()
                if not np.isfinite(loss):
                    raise RuntimeError(f"Nonfinite loss at image {i}")
                row = dict(index=i, weather=record[2], source_dataset=record[3], scene_id=record[5],
                    input_path=str(record[0]), target_path=str(record[1]), width=degraded.shape[1],
                    height=degraded.shape[0], loss=loss, inference_seconds=seconds)
                for prefix, image in (("input_", degraded), ("restored_", restored)):
                    psnr, ssim = image_metrics(image, target)
                    error = image-target
                    mse = float(np.mean(error.astype(np.float64)**2))
                    row.update({prefix+"psnr":psnr, prefix+"ssim":ssim,
                                prefix+"mae":float(np.mean(np.abs(error))), prefix+"mse":mse,
                                prefix+"rmse":float(np.sqrt(mse))})
                row["psnr_gain_db"] = row["restored_psnr"]-row["input_psnr"]
                row["ssim_gain"] = row["restored_ssim"]-row["input_ssim"]
                for threshold in THRESHOLDS:
                    row.update({f"mask_{threshold:.2f}_{key}":v for key,v in mask_counts(degraded,restored,target,threshold).items()})
                writer.writerow(row); stream.flush(); rows.append(row)
                if (i+1) % 25 == 0 or i+1 == len(dataset):
                    elapsed = time.perf_counter()-begin
                    rate = (i+1-start_index)/elapsed
                    status.update(state="running", completed=i+1, total=len(dataset), updated_at=stamp(),
                        elapsed_seconds=elapsed, images_per_second=rate,
                        remaining_seconds=(len(dataset)-i-1)/rate)
                    write_json(output/"status.json", status)
                    print(f"Test {i+1}/{len(dataset)} | {rate:.2f} images/s | ETA {status['remaining_seconds']/60:.1f} min", flush=True)
        status.update(state="creating_report", completed=len(rows)); write_json(output/"status.json",status)
        summary = summarize(rows)
        write_json(output/"summary.json", dict(provenance=provenance, groups=summary))
        overall = summary["overall"]
        write_json(output/"final_test.json", dict(loss=overall["mean_loss"], psnr=overall["restored_psnr"], ssim=overall["restored_ssim"]))
        make_report(output, rows, summary, provenance, model, dataset, device, pad_multiple)
        status.update(state="completed", finished_at=stamp(), report=str(output/"report.html"),
                      markdown_report=str(markdown_report_path(output)))
        write_json(output/"status.json", status)
        print("TESTING COMPLETED", json.dumps(overall), flush=True)
    except BaseException as error:
        status.update(state="failed", error=str(error), traceback=traceback.format_exc(), finished_at=stamp())
        write_json(output/"status.json", status)
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
