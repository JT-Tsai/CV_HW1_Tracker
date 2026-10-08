"""Temporal association and motion prediction for detector boxes."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .roi import Region


@dataclass
class Track:
    track_id: int
    class_id: int
    box: np.ndarray
    score_ema: float
    red_ratio: float = 0.0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    missed: int = 0
    hits: deque = field(default_factory=lambda: deque(maxlen=5))
    confirmed: bool = False


def box_center(box: np.ndarray) -> np.ndarray:
    return np.asarray(
        [(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5],
        dtype=np.float32,
    )


def center_distance_ratio(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Center distance normalized by the typical object box size."""
    distance = float(np.linalg.norm(box_center(box_a) - box_center(box_b)))
    size_a = max(float(np.hypot(box_a[2] - box_a[0], box_a[3] - box_a[1])), 1.0)
    size_b = max(float(np.hypot(box_b[2] - box_b[0], box_b[3] - box_b[1])), 1.0)
    return distance / max(0.5 * (size_a + size_b), 1.0)


def iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x0 = max(float(box_a[0]), float(box_b[0]))
    y0 = max(float(box_a[1]), float(box_b[1]))
    x1 = min(float(box_a[2]), float(box_b[2]))
    y1 = min(float(box_a[3]), float(box_b[3]))
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    return intersection / max(area_a + area_b - intersection, 1e-6)


class TemporalTracker:
    def __init__(
        self,
        alpha: float,
        confirm_hits: int,
        confirm_window: int,
        max_missed: int,
        match_iou: float,
        max_center_distance: float,
        box_alpha: float,
    ):
        self.alpha = alpha
        self.confirm_hits = confirm_hits
        self.confirm_window = confirm_window
        self.max_missed = max_missed
        self.match_iou = match_iou
        self.max_center_distance = max_center_distance
        self.box_alpha = box_alpha
        self.tracks: list[Track] = []
        self.next_id = 1

    def visible(self) -> list[dict]:
        return [
            {
                "track_id": track.track_id,
                "class_id": track.class_id,
                "box": track.box.copy(),
                "score": track.score_ema,
                "red_ratio": track.red_ratio,
                "missed": track.missed,
            }
            for track in self.tracks
            if track.confirmed
        ]

    def predict(self, skip_prediction_classes: set[int] | None = None) -> list[dict]:
        """Advance tracks without counting a detector miss."""
        skip_prediction_classes = skip_prediction_classes or set()
        for track in self.tracks:
            if track.class_id not in skip_prediction_classes:
                track.box = (track.box + track.velocity).astype(np.float32)
            track.velocity *= 0.9
        return self.visible()

    def update(
        self,
        detections: list[dict],
        skip_prediction_classes: set[int] | None = None,
    ) -> list[dict]:
        skip_prediction_classes = skip_prediction_classes or set()
        unmatched_tracks = set(range(len(self.tracks)))
        unmatched_detections = set(range(len(detections)))
        matches = []
        candidates = []
        for track_index, track in enumerate(self.tracks):
            predicted = track.box + track.velocity
            for detection_index, detection in enumerate(detections):
                if track.class_id != detection["class_id"]:
                    continue
                overlap = iou(predicted, detection["box"])
                distance = center_distance_ratio(predicted, detection["box"])
                if track.class_id == 0 and overlap < self.match_iou:
                    continue
                if overlap < self.match_iou and distance > self.max_center_distance:
                    continue
                association_score = overlap + max(
                    0.0, 1.0 - distance / self.max_center_distance
                ) * 0.25
                candidates.append((association_score, track_index, detection_index))

        for _, track_index, detection_index in sorted(candidates, reverse=True):
            if (
                track_index not in unmatched_tracks
                or detection_index not in unmatched_detections
            ):
                continue
            matches.append((track_index, detection_index))
            unmatched_tracks.remove(track_index)
            unmatched_detections.remove(detection_index)

        for track_index, detection_index in matches:
            track = self.tracks[track_index]
            detection = detections[detection_index]
            new_box = detection["box"]
            predicted = track.box + track.velocity
            measured_velocity = new_box - track.box
            track.velocity = 0.7 * track.velocity + 0.3 * measured_velocity
            track.box = (
                self.box_alpha * new_box
                + (1.0 - self.box_alpha) * predicted
            ).astype(np.float32)
            track.score_ema = (
                self.alpha * detection["score"]
                + (1 - self.alpha) * track.score_ema
            )
            track.red_ratio = (
                self.alpha * detection.get("red_ratio", 0.0)
                + (1 - self.alpha) * track.red_ratio
            )
            track.missed = 0
            track.hits.append(1)
            track.confirmed = sum(track.hits) >= self.confirm_hits

        for track_index in unmatched_tracks:
            track = self.tracks[track_index]
            if track.class_id not in skip_prediction_classes:
                track.box = track.box + track.velocity
            track.score_ema *= 1 - 0.5 * (1 - self.alpha)
            track.missed += 1
            track.hits.append(0)

        for detection_index in unmatched_detections:
            detection = detections[detection_index]
            track = Track(
                track_id=self.next_id,
                class_id=detection["class_id"],
                box=detection["box"].copy(),
                score_ema=detection["score"],
                red_ratio=detection.get("red_ratio", 0.0),
                hits=deque([1], maxlen=self.confirm_window),
            )
            track.confirmed = self.confirm_hits <= 1
            self.next_id += 1
            self.tracks.append(track)

        self.tracks = [
            track for track in self.tracks if track.missed <= self.max_missed
        ]
        return self.visible()

    def regions(self, padding: int, width: int, height: int) -> list[Region]:
        return [
            Region(
                max(0, int(track.box[0]) - padding),
                max(0, int(track.box[1]) - padding),
                min(width, int(track.box[2]) + padding),
                min(height, int(track.box[3]) + padding),
            )
            for track in self.tracks
        ]
