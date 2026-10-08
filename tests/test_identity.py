import numpy as np

from car_detection.tracking.identity import classify_car


def test_classify_yellow_blue_car():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    frame[:, :10] = (255, 0, 0)
    frame[:, 10:] = (0, 220, 220)
    identity, confidence, features = classify_car(frame, [0, 0, 40, 40])
    assert identity == 1
    assert confidence > 0
    assert features["blue_ratio"] > 0


def test_classify_red_blue_car():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    frame[:, :10] = (255, 0, 0)
    frame[:, 10:] = (0, 0, 220)
    identity, _, _ = classify_car(frame, [0, 0, 40, 40])
    assert identity == 2
