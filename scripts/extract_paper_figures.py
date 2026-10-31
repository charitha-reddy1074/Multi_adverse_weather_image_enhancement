"""Render faithful crops of AIRFormer figures for the local comparison README.

The source PDF is never modified. Coordinates are measured in PDF page points
from the top-left; the original figure labels/arrows/captions are preserved.
Requires requirements-audit.txt. No generated/redrawn architecture is substituted.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pypdfium2 as pdfium


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", type=Path)
    parser.add_argument("--output", type=Path, default=Path("docs/architecture/figures"))
    args = parser.parse_args()
    matches = list(Path(".").glob("Frequency-Oriented*.pdf"))
    source = args.paper or (matches[0] if len(matches) == 1 else None)
    if source is None:
        parser.error("Supply --paper when exactly one matching root PDF is unavailable")
    args.output.mkdir(parents=True, exist_ok=True)
    figures = {
        "figure4_airformer.png": (3, (45, 242, 567, 483), "Figure 4, printed page 1888"),
        "figure5_queried_attention.png": (5, (45, 42, 567, 303), "Figure 5, printed page 1890"),
        "table13_efficiency.png": (10, (313, 282, 567, 403), "Table XIII, printed page 1895"),
        "equations11_13.png": (6, (313, 268, 568, 655), "Equations 11-13, printed page 1891"),
    }
    metadata = {"source_pdf": source.name,
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "render_scale": 3, "coordinate_origin": "top-left, PDF points", "crops": {}}
    with pdfium.PdfDocument(str(source)) as document:
        for name, (page_number, bounds, description) in figures.items():
            page = document[page_number - 1]
            rendered = page.render(scale=3).to_pil()
            crop = rendered.crop(tuple(round(value * 3) for value in bounds))
            crop.save(args.output / name)
            metadata["crops"][name] = dict(pdf_page=page_number, bounds=list(bounds),
                                            description=description, pixels=list(crop.size))
    (args.output / "provenance.json").write_text(json.dumps(metadata, indent=2), encoding="utf8")
    print(f"Extracted {len(figures)} faithful paper crops to {args.output}")


if __name__ == "__main__":
    main()
