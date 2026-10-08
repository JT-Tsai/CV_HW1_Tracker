#!/usr/bin/env python3
"""Runtime loop for filtered-ROI YOLO detection."""

from __future__ import annotations

import time

import cv2
import numpy as np

from ..config import RealtimeConfig
from .inference import predict_regions, suppress_overlaps
from .roi import Region, select_regions_bgr, square_region
from .tracker import TemporalTracker


def parse_source(source: str):
    return int(source) if source.isdigit() else source


def merge_regions(regions: list[Region]) -> list[Region]:
    output: list[Region] = []
    for region in regions:
        merged = False
        for index, existing in enumerate(output):
            if not (region.x1 < existing.x0 or existing.x1 < region.x0
                    or region.y1 < existing.y0 or existing.y1 < region.y0):
                output[index] = Region(
                    min(region.x0, existing.x0), min(region.y0, existing.y0),
                    max(region.x1, existing.x1), max(region.y1, existing.y1),
                )
                merged = True
                break
        if not merged:
            output.append(region)
    return output


def red_pixel_ratio(frame: np.ndarray, box: np.ndarray) -> float:
    """Estimate how much of a detection crop is genuinely red."""
    height, width = frame.shape[:2]
    x0, y0, x1, y1 = np.rint(box).astype(int)
    x0, x1 = max(0, x0), min(width, x1)
    y0, y1 = max(0, y0), min(height, y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    crop = frame[y0:y1, x0:x1]
    blue, green, red = crop[..., 0], crop[..., 1], crop[..., 2]
    mask = (
        (red > 70)
        & (red > green + 20)
        & (red > blue + 20)
        & (red > green * 1.15)
    )
    return float(mask.mean())


def detect_rois(
    model,
    frame: np.ndarray,
    regions: list[Region],
    args: RealtimeConfig,
    allowed_classes: set[int] | None = None,
) -> list[dict]:
    detections = predict_regions(
        model,
        frame,
        regions,
        image_size=args.imgsz,
        confidence=min(args.conf, args.red_cone_conf, args.toy_car_conf),
        device=args.device,
        convert_to_rgb=True,
    )
    filtered = []
    for detection in detections:
        class_id = detection["class_id"]
        if allowed_classes is not None and class_id not in allowed_classes:
            continue
        red_ratio = 0.0
        if class_id == 0:
            red_ratio = red_pixel_ratio(frame, detection["box"])
            # Both pipelines require color evidence for red cones. This
            # prevents a toy car from spawning a class-0 track.
            if detection["score"] < args.red_cone_conf:
                continue
            if (
                args.strict_red_color and red_ratio < args.red_color_min
            ):
                continue
            if not args.strict_red_color:
                if (
                    detection["score"] < args.red_cone_high_conf
                    and red_ratio < args.red_color_min
                ):
                    continue
        elif class_id in (1, 2) and detection["score"] < args.toy_car_conf:
            continue
        detection["red_ratio"] = red_ratio
        filtered.append(detection)

    # Overlapping ROIs can return the same object more than once. Suppress
    # duplicate boxes before tracking, otherwise one object can receive two IDs.
    return suppress_overlaps(filtered, same_class_only=True)


def frame_motion_ratio(
    frame: np.ndarray,
    previous_gray: np.ndarray | None,
    pixel_threshold: int,
    dilation: int,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Return blurred grayscale frame and changed-pixel ratio."""
    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    if previous_gray is None:
        return gray, 1.0, np.full_like(gray, 255)
    difference = cv2.absdiff(previous_gray, gray)
    motion = cv2.threshold(
        difference, pixel_threshold, 255, cv2.THRESH_BINARY
    )[1]
    motion = cv2.dilate(motion, np.ones((dilation, dilation), np.uint8))
    return gray, float(np.mean(motion > 0)), motion


def motion_regions(
    motion_mask: np.ndarray | None,
    width: int,
    height: int,
    min_area: int,
    padding: int,
) -> list[Region]:
    """Convert large frame-difference components into detector ROIs."""
    if motion_mask is None:
        return []
    contours, _ = cv2.findContours(
        motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    regions = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        regions.append(
            Region(
                max(0, x - padding),
                max(0, y - padding),
                min(width, x + w + padding),
                min(height, y + h + padding),
            )
        )
    return regions


def max_motion_in_regions(
    motion_mask: np.ndarray | None,
    regions: list[Region],
) -> float:
    """Return the largest changed-pixel ratio inside any ROI."""
    if motion_mask is None or not regions:
        return 0.0
    height, width = motion_mask.shape[:2]
    ratios = []
    for region in regions:
        x0 = max(0, min(width, region.x0))
        y0 = max(0, min(height, region.y0))
        x1 = max(0, min(width, region.x1))
        y1 = max(0, min(height, region.y1))
        if x1 > x0 and y1 > y0:
            ratios.append(float(np.mean(motion_mask[y0:y1, x0:x1] > 0)))
    return max(ratios, default=0.0)


def filter_visible_tracks(
    detections: list[dict],
    args: RealtimeConfig,
    frame: np.ndarray | None = None,
) -> list[dict]:
    """Apply the red-color rule to persistent tracks as well as detections."""
    visible = []
    for detection in detections:
        # A track may be retained internally for re-association, but it is
        # not drawn when the current detector did not match it.
        if detection.get("missed", 0) > 0:
            continue
        if detection["class_id"] == 0 and detection["score"] < args.red_cone_conf:
            continue
        if detection["class_id"] in (1, 2) and detection["score"] < args.toy_car_conf:
            continue
        current_red_ratio = detection.get("red_ratio", 0.0)
        if frame is not None and detection["class_id"] == 0:
            current_red_ratio = red_pixel_ratio(frame, detection["box"])
        if (
            detection["class_id"] == 0
            and args.strict_red_color
            and current_red_ratio < args.red_color_min
        ):
            continue
        visible.append(detection)
    return visible


def run_realtime(
    args: RealtimeConfig,
    model,
    capture,
    resources: dict | None = None,
) -> None:
    """Process frames with already-open model and capture resources.

    The CLI owns resource lifetime. ``resources`` lets the CLI release a video
    writer even when frame processing raises an exception.
    """
    resources = resources if resources is not None else {}

    def make_tracker() -> TemporalTracker:
        tracker = TemporalTracker(
            args.ema_alpha, args.confirm_hits, args.confirm_window,
            args.max_missed, args.match_iou,
            args.max_center_distance, args.box_alpha,
        )
        if args.temporal_mode == "ema":
            tracker.confirm_hits = 1
            tracker.max_missed = 0
        return tracker

    # Keep object classes independent: red cones are found from color ROIs,
    # while toy cars are found only from frame-difference ROIs.
    red_tracker = make_tracker()
    car_tracker = make_tracker()
    writer = None
    resources["writer"] = None
    frame_index = 0
    detector_calls = 0
    detector_regions = 0
    detected_frames = 0
    visible_detections = 0
    motion_skips = 0
    timing = {"loop": 0.0, "roi": 0.0, "detect": 0.0, "track": 0.0}
    previous_gray = None
    last_red_regions: list[Region] = []

    while True:
        loop_start = time.perf_counter()
        ok, frame = capture.read()
        if not ok:
            break
        height, width = frame.shape[:2]
        motion_changed = True
        current_gray = None
        motion_mask = None
        if args.motion_gate or args.motion_roi or args.temporal_mode == "tracker":
            current_gray, motion_ratio, motion_mask = frame_motion_ratio(
                frame,
                previous_gray,
                args.motion_pixel_threshold,
                args.motion_dilation,
            )
            previous_gray = current_gray
            if args.motion_gate:
                motion_changed = motion_ratio >= args.motion_threshold
        if writer is None and args.output_video is not None:
            fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
            args.output_video.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                str(args.output_video), cv2.VideoWriter_fourcc(*"mp4v"),
                fps, (width, height),
            )
            resources["writer"] = writer

        roi_start = time.perf_counter()
        red_regions: list[Region] = []
        if args.filter == "color":
            color_check_due = (
                not red_tracker.tracks
                or frame_index % args.red_cone_detect_every == 0
            )
            if color_check_due:
                _, last_red_regions = select_regions_bgr(
                    frame,
                    padding=args.filter_padding,
                    min_area=args.filter_min_area,
                    dilation=args.filter_dilation,
                )
            red_regions = list(last_red_regions)
        car_regions = (
            motion_regions(
                motion_mask,
                width,
                height,
                args.motion_roi_min_area,
                args.motion_roi_padding,
            )
            if args.motion_roi else []
        )
        timing["roi"] += time.perf_counter() - roi_start

        # A cone can remain still for many frames. Keep its existing box and
        # ID, and only rerun cone detection on a periodic check or local motion.
        if args.temporal_mode != "none" and red_tracker.tracks:
            if not red_regions:
                red_regions = red_tracker.regions(args.filter_padding, width, height)
        red_regions = merge_regions(red_regions)
        red_regions = [square_region(region, width, height, minimum=512) for region in red_regions]
        red_motion_probe = (
            red_tracker.regions(32, width, height)
            if red_tracker.tracks else red_regions
        )
        red_update_due = (
            args.filter == "color"
            and (
                not red_tracker.tracks
                or frame_index % args.red_cone_detect_every == 0
                or max_motion_in_regions(motion_mask, red_motion_probe)
                >= args.red_cone_motion_threshold
            )
        )
        if red_update_due:
            detect_start = time.perf_counter()
            red_detections = detect_rois(
                model, frame, red_regions, args, allowed_classes={0}
            ) if red_regions else []
            timing["detect"] += time.perf_counter() - detect_start
            detector_calls += bool(red_regions)
            detector_regions += len(red_regions)
            detected_frames += bool(red_regions)
            red_visible = (
                red_detections
                if args.temporal_mode == "none"
                else (
                    red_tracker.update(red_detections)
                    if red_detections
                    else red_tracker.predict(skip_prediction_classes={0})
                )
            )
        elif args.temporal_mode == "none":
            red_visible = []
        else:
            red_visible = red_tracker.predict(skip_prediction_classes={0})

        # Toy cars are driven by frame difference only. With detect-every=1,
        # every frame containing a changed car ROI gets a fresh YOLO update.
        car_regions = merge_regions(car_regions)
        if args.temporal_mode != "none" and not car_regions and car_tracker.tracks:
            # No difference means the car is unchanged; keep its last ID/box.
            car_visible = car_tracker.predict(skip_prediction_classes={1})
            motion_skips += 1
        elif frame_index % args.detect_every != 0:
            car_visible = car_tracker.visible() if args.temporal_mode != "none" else []
            motion_skips += 1
        else:
            car_regions = [square_region(region, width, height, minimum=512) for region in car_regions]
            detect_start = time.perf_counter()
            car_detections = detect_rois(
                model, frame, car_regions, args, allowed_classes={1}
            ) if car_regions else []
            timing["detect"] += time.perf_counter() - detect_start
            detector_calls += bool(car_regions)
            detector_regions += len(car_regions)
            detected_frames += bool(car_regions)
            if args.temporal_mode == "none":
                car_visible = car_detections
            else:
                car_visible = (
                    car_tracker.update(car_detections)
                    if car_detections
                    else car_tracker.predict(skip_prediction_classes={1})
                )

        regions = red_regions + car_regions
        visible = filter_visible_tracks(red_visible + car_visible, args, frame)
        visible_detections += len(visible)

        for detection in visible:
            x0, y0, x1, y1 = detection["box"].astype(int)
            color = (0, 180, 0) if detection["class_id"] == 1 else (0, 0, 255)
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 3)
            label = f"class={detection['class_id']} {detection['score']:.2f}"
            if "track_id" in detection:
                label = f"id={detection['track_id']} {label}"
            cv2.putText(frame, label, (x0, max(24, y0 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if args.draw_roi:
            for region in regions:
                cv2.rectangle(frame, (region.x0, region.y0), (region.x1, region.y1), (255, 200, 0), 1)

        if writer is not None:
            writer.write(frame)
        if args.display:
            cv2.imshow("YOLO filtered ROI", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        frame_index += 1
        timing["loop"] += time.perf_counter() - loop_start
        if args.max_frames is not None and frame_index >= args.max_frames:
            break

    if writer is not None:
        writer.release()
        resources["writer"] = None
    if args.profile and frame_index:
        print(f"profile_frames={frame_index}")
        print(f"profile_detector_calls={detector_calls}")
        print(f"profile_detector_regions={detector_regions}")
        print(f"profile_detected_frames={detected_frames}")
        print(f"profile_visible_detections={visible_detections}")
        print(f"profile_motion_skips={motion_skips}")
        print(f"profile_pipeline_fps={frame_index / max(timing['loop'], 1e-9):.2f}")
        print(f"profile_roi_ms_per_detect={timing['roi'] / max(detected_frames, 1) * 1000:.2f}")
        print(f"profile_detect_ms_per_detect={timing['detect'] / max(detected_frames, 1) * 1000:.2f}")
