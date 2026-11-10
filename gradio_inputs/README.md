# Images for the Gradio demonstration

Open http://127.0.0.1:7860 and select an example, or upload one of these files.
Choose **Full image**, then click **Enhance image**.

| Weather | Held-out test input | Web input |
|---|---|---|
| Rain streaks | `test_dataset/rain.png` | `web/rain.png` |
| Raindrops | `test_dataset/raindrop.png` | `web/raindrop.png` |
| Snow | `test_dataset/snow.png` | `web/snow.png` |
| Haze | `test_dataset/haze.png` | `web/haze.png` |

Use the **test_dataset** images first when demonstrating visible restoration.
They are selected favorable examples from the custom held-out test manifest,
even where original archive paths contain `train` or `training`.
Selection details are in `test_dataset/provenance.json`.

The **web** photos are additional real-world tests, with no paired clean
reference. They do not guarantee successful restoration. Web inputs were only
resized and converted to PNG; no enhancement filters or synthetic weather were
applied. Original sources, authors, licenses, and changes are recorded in
`web/ATTRIBUTION.md` and `web/sources.json`.

Model outputs and comparisons for these eight inputs are saved separately in
`outputs/gradio_demonstration/` after the demonstration verification run.

## Verified presentation results

All eight images completed full-image inference and saved PNG outputs.
The four selected test examples show visible restoration. On the web photos,
rain and snow show modest changes, the large glass droplets largely remain,
and the hazy scene becomes darker while substantial haze remains. Use these
web cases to show current real-world limitations, rather than claim strong
weather removal. Pixel changes alone do not establish improved quality.

Ready-to-show sheets:

- `../outputs/gradio_demonstration/test_dataset/all_weather_comparison.png`
- `../outputs/gradio_demonstration/web/all_weather_comparison.png`

Each sheet has input images on the left and actual model outputs on the right.
Individual weather comparisons and output PNGs are saved beside each sheet.
