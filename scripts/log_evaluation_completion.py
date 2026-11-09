"""Append an evaluation's final outcome to docs/LOG.md when it becomes terminal."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = (root/args.output).resolve()
    recorded = output/"completion_logged.json"
    while True:
        status = json.loads((output/"status.json").read_text(encoding="utf8"))
        if status.get("state") in {"completed", "failed"}:
            identity = {k:status.get(k) for k in ("state", "started_at", "finished_at")}
            if recorded.exists() and json.loads(recorded.read_text(encoding="utf8")) == identity:
                return
            stamp = datetime.now().astimezone().isoformat(timespec="seconds")
            lines = [f"\n\n## {stamp} - Full model testing {status['state']}\n"]
            lines.append(f"- Evaluation output: `{output.relative_to(root).as_posix()}`. Status evidence: `status.json`; logs: `evaluation.stdout.log` and `evaluation.stderr.log`.")
            if status["state"] == "completed":
                overall = json.loads((output/"summary.json").read_text(encoding="utf8"))["groups"]["overall"]
                lines.append(f"- All {overall['images']:,} test images evaluated. Mean restored PSNR: {overall['restored_psnr']:.4f} dB; SSIM: {overall['restored_ssim']:.6f}; mean PSNR gain: {overall['mean_psnr_gain_db']:+.4f} dB.")
                lines.append("- Final metrics, auxiliary pixel-mask scores (not weather classification), and visualizations are available in `summary.json`, `final_test.json`, `report.html`, and `visualizations/` within the evaluation output. The Markdown report is under `docs/reports/`.")
            else:
                lines.append(f"- Evaluation stopped with error: {status.get('error', 'see status traceback')}. Saved per-image results remain available; this does not discard the epoch-20 training checkpoint.")
            with (root/"docs"/"LOG.md").open("a", encoding="utf8") as stream:
                stream.write("\n".join(lines)+"\n")
            recorded.write_text(json.dumps(identity, indent=2), encoding="utf8")
            print(f"Logged evaluation {status['state']}", flush=True)
            return
        if args.once:
            print(f"Evaluation is {status.get('state')}; no final outcome logged", flush=True)
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
