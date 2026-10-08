from car_detection.tracking.motion import MotionEstimator


def test_motion_estimator_reports_forward_motion():
    estimator = MotionEstimator(min_span_seconds=0.01, min_speed_mm_s=1)
    estimator.update(1, 0, [0, 0])
    state = estimator.update(1, 100_000, [10, 0])
    assert state["speed_mm_s"] == 100.0
    assert state["movement_direction_deg"] == 0.0
