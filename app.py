"""Small Gradio frontend for the trained weather restoration model."""

import argparse
import logging
import time
from pathlib import Path
from threading import Lock
from uuid import uuid4

import gradio as gr
import torch
from PIL import ImageOps

from utils.inference import load_model, restore_image

ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "checkpoints/four_weather_resume_epoch17/latest.pt"
INPUTS_DIR = ROOT / "gradio_inputs"
LOGGER = logging.getLogger(__name__)


class RestorationDemo:
    """Load once on first use and serialize access to the model."""

    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, device="auto"):
        self.checkpoint = Path(checkpoint).resolve()
        self.requested_device = device
        self.model = None
        self.device = None
        self.lock = Lock()
        self.output_dir = ROOT / "outputs/gradio"

    def restore(self, image, tile_size):
        if image is None:
            raise gr.Error("Upload an image first.")
        image = ImageOps.exif_transpose(image).convert("RGB")
        with self.lock:
            started = time.perf_counter()
            try:
                if self.model is None:
                    if not self.checkpoint.is_file():
                        raise gr.Error(
                            "Checkpoint not found. Start app.py with --checkpoint path/to/model.pt."
                        )
                    self.model, self.device = load_model(
                        self.checkpoint, self.requested_device
                    )
                restored, elapsed = restore_image(
                    self.model, image, self.device, tile_size=int(tile_size)
                )
                self.output_dir.mkdir(parents=True, exist_ok=True)
                destination = self.output_dir / f"restored_{uuid4().hex}.png"
                restored.save(destination)
            except gr.Error:
                raise
            except torch.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise gr.Error(
                    "Not enough device memory. Choose a smaller tile size or upload a smaller image."
                ) from None
            except Exception:
                LOGGER.exception("Image restoration failed")
                raise gr.Error(
                    "Restoration failed. Check the terminal for details and verify the checkpoint/device."
                ) from None
            total = time.perf_counter() - started
            status = (
                f"Completed · {image.width} × {image.height} px · {self.device}\n"
                f"Model execution: {elapsed:.3f} s · Total: {total:.3f} s\n"
                "Total includes loading the model on the first run and saving the PNG."
            )
            return restored, str(destination), status


def create_app(checkpoint=DEFAULT_CHECKPOINT, device="auto"):
    runner = RestorationDemo(checkpoint, device)
    with gr.Blocks(title="Weather Image Restoration", analytics_enabled=False) as demo:
        gr.Markdown(
            "# Weather Image Restoration\n"
            "Upload a photo affected by **rain, raindrops, snow, or haze**, "
            "then select **Enhance image**."
        )
        with gr.Row():
            source = gr.Image(type="pil", image_mode="RGB", label="Input image", height=360)
            result = gr.Image(
                type="pil", label="Enhanced image", format="png", interactive=False, height=360
            )
        with gr.Accordion("Processing options", open=False):
            tile_size = gr.Dropdown(
                choices=[("Full image", 0), ("256 px", 256), ("512 px", 512)],
                value=0,
                label="Tile size",
                info="Smaller tiles use less memory. Tiling can affect the result near tile boundaries.",
            )
            gr.Markdown(f"Checkpoint: `{Path(checkpoint).name}` · Device: `{device}`")
        with gr.Row():
            run = gr.Button("Enhance image", variant="primary")
            clear = gr.ClearButton(value="Clear")
        status = gr.Textbox(label="Execution status", interactive=False, lines=3)
        download = gr.File(label="Download enhanced PNG", interactive=False)
        gr.Markdown(
            "### Images for your demonstration\n"
            "Select an image below or upload a file from **gradio_inputs**, "
            "then click **Enhance image**. "
            "These are selected strong examples from the held-out test split; "
            "results on other images vary. Examples use full-image processing."
        )
        gr.Examples(
            examples=[
                [str(INPUTS_DIR / "test_dataset" / f"{weather}.png"), 0]
                for weather in ("rain", "raindrop", "snow", "haze")
            ],
            inputs=[source, tile_size],
            example_labels=["Rain", "Raindrops", "Snow", "Haze"],
            label="Test dataset: rain, raindrops, snow, haze",
            cache_examples=False,
        )
        gr.Examples(
            examples=[
                [str(INPUTS_DIR / "web" / f"{weather}.png"), 0]
                for weather in ("rain", "raindrop", "snow", "haze")
            ],
            inputs=[source, tile_size],
            example_labels=["Web rain", "Web raindrops", "Web snow", "Web haze"],
            label="Web photos: real-world testing",
            cache_examples=False,
        )
        with gr.Accordion("Web photo credits", open=False):
            gr.Markdown((INPUTS_DIR / "web" / "ATTRIBUTION.md").read_text(encoding="utf8"))
        source.change(
            lambda: (None, None, ""),
            outputs=[result, download, status],
            queue=False,
            api_name=False,
        )
        run.click(
            runner.restore,
            inputs=[source, tile_size],
            outputs=[result, download, status],
            concurrency_limit=1,
            api_name="restore",
        )
        clear.add([source, result, download, status])
    return demo.queue(max_size=8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    create_app(args.checkpoint, args.device).launch(
        server_name="127.0.0.1", server_port=args.port, share=False
    )


if __name__ == "__main__":
    main()
