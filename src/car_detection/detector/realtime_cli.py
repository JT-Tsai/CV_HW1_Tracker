"""Command-line interface for the standalone detector."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from ..config import (
    ConfigError,
    REALTIME_DEFAULTS,
    TRACKER_DEFAULTS,
    RealtimeConfig,
    validate_realtime_config,
)
from ..device import resolve_device
from .realtime_runner import parse_source, run_realtime


def parse_args(argv=None) -> RealtimeConfig:
    parser = argparse.ArgumentParser(
        description="Run real-time filtered-ROI YOLO detection."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--source", required=True, help="Camera index, image, or video path."
    )
    parser.add_argument("--output-video", type=Path, default=None)
    parser.add_argument(
        "--device",
        default=REALTIME_DEFAULTS.device,
        help="auto, CUDA index such as 0, mps, or cpu.",
    )
    parser.add_argument("--imgsz", type=int, default=REALTIME_DEFAULTS.image_size)
    parser.add_argument("--conf", type=float, default=REALTIME_DEFAULTS.confidence)
    parser.add_argument(
        "--red-cone-conf",
        type=float,
        default=REALTIME_DEFAULTS.red_cone_confidence,
        help="Minimum confidence for red-cone candidates before color filtering.",
    )
    parser.add_argument(
        "--red-cone-high-conf",
        type=float,
        default=REALTIME_DEFAULTS.red_cone_high_confidence,
        help="High-confidence red-cone detections allowed without color evidence.",
    )
    parser.add_argument(
        "--red-color-min",
        type=float,
        default=REALTIME_DEFAULTS.red_color_minimum,
        help="Minimum red-pixel ratio for low-confidence red-cone detections.",
    )
    parser.add_argument(
        "--toy-car-conf",
        type=float,
        default=REALTIME_DEFAULTS.toy_car_confidence,
    )
    parser.add_argument("--filter", choices=("color", "none"), default="color")
    parser.add_argument(
        "--filter-padding", type=int, default=REALTIME_DEFAULTS.filter_padding
    )
    parser.add_argument(
        "--filter-min-area", type=int, default=REALTIME_DEFAULTS.filter_min_area
    )
    parser.add_argument(
        "--filter-dilation", type=int, default=REALTIME_DEFAULTS.filter_dilation
    )
    parser.add_argument(
        "--temporal-mode",
        choices=("none", "ema", "tracker"),
        default=REALTIME_DEFAULTS.temporal_mode,
    )
    parser.add_argument("--ema-alpha", type=float, default=TRACKER_DEFAULTS.ema_alpha)
    parser.add_argument(
        "--confirm-hits", type=int, default=TRACKER_DEFAULTS.confirm_hits
    )
    parser.add_argument(
        "--confirm-window", type=int, default=TRACKER_DEFAULTS.confirm_window
    )
    parser.add_argument("--max-missed", type=int, default=TRACKER_DEFAULTS.max_missed)
    parser.add_argument("--match-iou", type=float, default=TRACKER_DEFAULTS.match_iou)
    parser.add_argument(
        "--max-center-distance",
        type=float,
        default=TRACKER_DEFAULTS.max_center_distance,
        help="Maximum center movement relative to object size for ID matching.",
    )
    parser.add_argument(
        "--box-alpha",
        type=float,
        default=TRACKER_DEFAULTS.box_alpha,
        help="Detection box smoothing weight for confirmed tracks.",
    )
    parser.add_argument(
        "--motion-gate",
        action="store_true",
        help="Use frame difference to skip YOLO when the scene is unchanged.",
    )
    parser.add_argument(
        "--motion-threshold",
        type=float,
        default=REALTIME_DEFAULTS.motion_threshold,
        help="Minimum changed-pixel ratio required to run YOLO.",
    )
    parser.add_argument(
        "--motion-pixel-threshold",
        type=int,
        default=REALTIME_DEFAULTS.motion_pixel_threshold,
    )
    parser.add_argument(
        "--motion-dilation", type=int, default=REALTIME_DEFAULTS.motion_dilation
    )
    motion_roi = parser.add_mutually_exclusive_group()
    motion_roi.add_argument(
        "--motion-roi",
        dest="motion_roi",
        action="store_true",
        default=REALTIME_DEFAULTS.motion_roi,
        help="Use large frame-difference components as YOLO ROIs (default).",
    )
    motion_roi.add_argument(
        "--no-motion-roi",
        dest="motion_roi",
        action="store_false",
        help="Disable motion-based car ROIs.",
    )
    parser.add_argument(
        "--motion-roi-min-area",
        type=int,
        default=REALTIME_DEFAULTS.motion_roi_min_area,
    )
    parser.add_argument(
        "--motion-roi-padding",
        type=int,
        default=REALTIME_DEFAULTS.motion_roi_padding,
    )
    parser.add_argument(
        "--detect-every",
        type=int,
        default=REALTIME_DEFAULTS.detect_every,
        help="Run YOLO every N frames; tracker predicts the intervening frames.",
    )
    parser.add_argument(
        "--red-cone-detect-every",
        type=int,
        default=REALTIME_DEFAULTS.red_cone_detect_every,
        help="Red-cone verification interval; motion in a color ROI triggers an earlier update.",
    )
    parser.add_argument(
        "--red-cone-motion-threshold",
        type=float,
        default=REALTIME_DEFAULTS.red_cone_motion_threshold,
        help="Changed-pixel ratio inside a red-cone ROI that triggers an early update.",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--draw-roi", action="store_true")
    parser.add_argument("--profile", action="store_true")

    args = RealtimeConfig.from_namespace(parser.parse_args(argv))
    if not args.model.is_file():
        parser.error(f"YOLO model not found: {args.model}")
    try:
        validate_realtime_config(args)
    except ConfigError as exc:
        parser.error(str(exc))
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install Ultralytics: pip install ultralytics") from exc

    args.device = resolve_device(args.device)
    model = YOLO(str(args.model))
    capture = cv2.VideoCapture(parse_source(args.source))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open source: {args.source}")

    resources = {}
    try:
        run_realtime(args, model, capture, resources)
    finally:
        capture.release()
        writer = resources.get("writer")
        if writer is not None:
            writer.release()
        if args.display:
            cv2.destroyAllWindows()
