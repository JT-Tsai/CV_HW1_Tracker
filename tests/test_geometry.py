import numpy as np

from car_detection.geometry.floor import FloorCalibration, transform_pixels


def test_transform_pixels_identity_matrix():
    points = transform_pixels([[10, 20], [30, 40]], np.eye(3))
    np.testing.assert_allclose(points, [[10, 20], [30, 40]])


def test_floor_calibration_maps_reference_corners():
    calibration = FloorCalibration.from_corners(
        [[0, 0], [100, 0], [0, 100], [100, 100]],
        image_size=(101, 101),
        tile_width_mm=600,
        tile_height_mm=600,
        cols=1,
        rows=1,
    )
    np.testing.assert_allclose(calibration.pixel_to_world(100, 100), (600, 600))
