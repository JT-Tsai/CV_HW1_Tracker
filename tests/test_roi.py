import numpy as np

from car_detection.detector.roi import select_regions_bgr


def test_roi_filter_finds_colorful_region():
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    frame[20:60, 20:60] = (0, 0, 255)
    _, regions = select_regions_bgr(frame, padding=0, min_area=20)
    assert regions
    assert any(region.x0 <= 20 and region.x1 >= 60 for region in regions)
