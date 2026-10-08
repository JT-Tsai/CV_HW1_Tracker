# Car Detection

This project calibrates a fixed camera to a tiled floor, detects toy cars with
YOLO, tracks them over time, and reports floor position and motion in
millimeters.

## Repository Structure

```text
.
├── src/car_detection/
│   ├── __main__.py              # `python -m car_detection` entry point
│   ├── cli.py                   # Main CLI wrapper
│   ├── pipeline.py              # Calibration, detection, tracking, output
│   ├── config.py                # Typed settings and validation
│   ├── calibration.py           # Snapshot and floor-grid selection
│   ├── rendering.py             # Camera and top-down visualizations
│   ├── geometry/floor.py        # Pixel-to-floor transformation
│   ├── tracking/                # Motion and vehicle identity logic
│   └── detector/                # ROI, YOLO, and temporal tracking
├── tests/                       # Deterministic unit tests
└── scripts/download_assets.sh   # Model/video downloader
```

## Conda Environment

Create and activate the development environment:

```bash
conda create -n car_vision python=3.10 -y
conda activate car_vision
pip install -r requirements.txt
pip install -e .
```

Open `scripts/download_assets.sh` and fill in the three Google Drive file IDs
near the top of the script. Then run the downloader:

```bash
bash scripts/download_assets.sh
```

`YOLO_MODEL_ID` downloads the standard `models/yolo/best.pt` checkpoint with
`red_cone` and `toy_car` classes.

## Implementation Overview

The main pipeline captures a frame, estimates the floor coordinate system from
clicked grid points, selects useful regions, runs YOLO inference, assigns
temporal tracks, and renders positions and motion. Shared inference and NMS
are in `detector/inference.py`. Runtime options belong in the typed dataclasses
and validators in `config.py`; workflow-specific code belongs in the pipeline
or realtime runner instead of being duplicated.

## CLI Commands

`-m` runs a package module, preserving the imports inside `car_detection`.

Run the calibrated pipeline from a camera:

```bash
python -m car_detection --source 0 --device auto --save-results
```

Run it on a video (All pipeline, `Do First` to check demo):

```bash
python -m car_detection \
  --source assets/videos/test2.mp4 \
  --device auto 
```

Run the standalone filtered-ROI detector:

```bash
python -m car_detection.detector.realtime \
  --model models/yolo/best.pt \
  --source assets/videos/test2.mp4 \
  --device auto \
  --max-frames 100
```

Motion-based car ROIs are enabled by default. Use `--no-motion-roi` to disable
them for diagnostics. Add `--output-video output/detection.mp4` to save the
standalone detector output.

## Image Calibration

Calibration requires a desktop session because OpenCV opens interactive
windows. Keep the camera resolution, crop, viewpoint, and floor setup fixed.
The default grid has `cols=2` and `rows=3`, so it expects a 3-by-4 grid of
floor intersections.

Start calibration with:

```bash
python -m car_detection \
  --source 0 \
  --cols 2 \
  --rows 3 \
  --tile-width-mm 600 \
  --tile-height-mm 600 \
  --save-results
```

1. Start the main CLI with a camera or video source.
2. Press `S` in the snapshot window to freeze one calibration image.
3. Click floor-grid intersections from left to right, row by row.
4. Press `C` to skip a hidden point, `R` to reset, and `Enter` to accept.
5. Press `Q` or `Esc` to cancel.

Use `--save-results` to store `calibration.json`, the calibration snapshot,
the rendered detection video, and `positions.jsonl` under
`onsite_sessions/<session-id>/`. Set `--tile-width-mm` and
`--tile-height-mm` to the measured floor-tile dimensions when available.

## Development

```bash
python -m compileall src scripts
python -m car_detection --help
python -m car_detection.detector.realtime --help
```
