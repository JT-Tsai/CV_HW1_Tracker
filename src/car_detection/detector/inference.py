"""YOLO inference helpers for the floor-detection pipeline."""

from __future__ import annotations

import time

import cv2
import numpy as np

from ..config import PipelineConfig
from .roi import Region, select_regions_bgr, square_region
from .tracker import iou


def predict_regions(
    model,
    frame: np.ndarray,
    regions: list[Region],
    *,
    image_size: int,
    confidence: float,
    device: str,
    classes: list[int] | tuple[int, ...] | None = None,
    convert_to_rgb: bool = False,
) -> list[dict]:
    """Run one shared YOLO prediction pass over image regions.

    ``convert_to_rgb`` is kept explicit because the two historical workflows
    intentionally use different Ultralytics input conventions.
    """
    if regions:
        crops = [frame[r.y0:r.y1, r.x0:r.x1] for r in regions]
        offsets = [(r.x0, r.y0) for r in regions]
    else:
        crops = [frame]
        offsets = [(0, 0)]
    if convert_to_rgb:
        crops = [cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) for crop in crops]

    results = model.predict(
        source=crops,
        imgsz=image_size,
        conf=confidence,
        classes=classes,
        device=device,
        verbose=False,
        stream=False,
    )
    detections = []
    for result, (x, y) in zip(results, offsets):
        if result.boxes is None:
            continue
        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy() + [x, y, x, y]
            detections.append(
                {
                    "box": xyxy.astype(np.float32),
                    "score": float(box.conf.item()),
                    "class_id": int(box.cls.item()),
                }
            )
    return detections


def suppress_overlaps(
    detections: list[dict],
    threshold: float = 0.5,
    same_class_only: bool = False,
) -> list[dict]:
    """Keep the highest-scoring box when overlapping regions duplicate it."""
    kept = []
    for detection in sorted(
        detections, key=lambda item: item["score"], reverse=True
    ):
        if not any(
            iou(detection["box"], item["box"]) > threshold
            for item in kept
            if not same_class_only or item["class_id"] == detection["class_id"]
        ):
            kept.append(detection)
    return kept


def detect(model, frame: np.ndarray, args: PipelineConfig) -> list[dict]:
    """Run YOLO on the highest-scoring color regions in one frame."""
    started = time.perf_counter()
    color_mask, regions = select_regions_bgr(frame, padding=args.roi_padding)
    height, width = color_mask.shape[:2]
    scores = [
        int(
            color_mask[
                max(0, region.y0):min(height, region.y1),
                max(0, region.x0):min(width, region.x1),
            ].sum()
        )
        for region in regions
    ]
    regions = [
        region
        for region, _ in sorted(
            zip(regions, scores), key=lambda item: item[1], reverse=True
        )[:args.max_roi_crops]
    ]
    regions = [
        square_region(region, frame.shape[1], frame.shape[0], args.min_roi_size)
        for region in regions
    ]
    prepared = time.perf_counter()
    detections = predict_regions(
        model,
        frame,
        regions,
        image_size=args.imgsz,
        confidence=args.conf,
        device=args.device,
        classes=args.vehicle_class,
    )

    args.last_detect_perf = {
        "roi_ms": (prepared - started) * 1000,
        "model_ms": (time.perf_counter() - prepared) * 1000,
        "crops": len(regions) if regions else 1,
    }
    return suppress_overlaps(detections)
