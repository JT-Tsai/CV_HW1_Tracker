import numpy as np

from car_detection.detector.tracker import TemporalTracker, iou


def test_iou_for_overlapping_boxes():
    value = iou(np.array([0, 0, 10, 10]), np.array([5, 5, 15, 15]))
    assert abs(value - 1 / 7) < 1e-6


def test_tracker_confirms_and_keeps_track_id():
    tracker = TemporalTracker(0.6, 1, 3, 2, 0.1, 2.0, 0.65)
    first = [{"box": np.array([0, 0, 10, 10], dtype=np.float32), "score": 0.9, "class_id": 1}]
    second = [{"box": np.array([1, 0, 11, 10], dtype=np.float32), "score": 0.9, "class_id": 1}]
    assert tracker.update(first)[0]["track_id"] == 1
    assert tracker.update(second)[0]["track_id"] == 1
