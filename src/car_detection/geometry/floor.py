"""Floor-coordinate utilities independent of YOLO and camera capture."""
import json
from pathlib import Path

import cv2
import numpy as np

# This module does not hard-code camera, model, or machine-specific paths.
# Keep it beside the application and import FloorCalibration directly.
# The caller supplies the calibration JSON path.


def is_ordered_boundary(points, minimum_area=100):
    """Return whether points form a consistently ordered convex boundary."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 4 or not np.isfinite(points).all():
        return False
    if abs(float(cv2.contourArea(points.astype(np.float32)))) <= minimum_area:
        return False
    edges = np.roll(points, -1, axis=0) - points
    next_edges = np.roll(edges, -1, axis=0)
    turns = edges[:, 0] * next_edges[:, 1] - edges[:, 1] * next_edges[:, 0]
    turns = turns[np.abs(turns) > 1e-8]
    return len(turns) > 0 and (np.all(turns > 0) or np.all(turns < 0))


def transform_pixels(points, matrix):
    """Transform image coordinates ``(u, v)`` to floor coordinates in mm."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError('The coordinate transform must be a finite 3x3 matrix')
    if not np.isfinite(points).all():
        raise ValueError('Image coordinates must be finite')
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    if np.any(np.abs(homogeneous[:, 2]) < 1e-12):
        raise ValueError('The image point is too close to the transform horizon')
    return homogeneous[:, :2] / homogeneous[:, 2, None]


class FloorCalibration:
    """Calibration for a fixed camera and a physical floor reference frame."""

    def __init__(self, metadata):
        self.metadata = dict(metadata)
        self.matrix = np.asarray(metadata['image_to_world_mm'], dtype=np.float64)
        if (
            self.matrix.shape != (3, 3)
            or not np.isfinite(self.matrix).all()
            or np.linalg.matrix_rank(self.matrix) < 3
        ):
            raise ValueError('Invalid floor calibration matrix')
        self.image_size = tuple(metadata['image_size'])  # (width, height)
        cols, rows = metadata['tile_count']
        self.width_mm = cols * float(metadata['tile_width_mm'])
        self.height_mm = rows * float(metadata['tile_height_mm'])
        if min(*self.image_size, self.width_mm, self.height_mm) <= 0:
            raise ValueError('Image and floor dimensions must be positive')

    @classmethod
    def from_corners(
        cls,
        corners_uv,
        image_size,
        tile_width_mm=600,
        tile_height_mm=600,
        cols=2,
        rows=3,
        measured=False,
        grid_indices=None,
    ):
        """Build calibration from a grid of points or legacy boundary points.

        The current interactive workflow uses the ``(cols + 1) * (rows + 1)``
        grid intersections in row-major order. ``grid_indices`` allows hidden
        intersections to be skipped. Four- and six-point boundary layouts
        remain supported for older calibration data.
        """
        points = np.asarray(corners_uv, dtype=np.float32)
        if min(tile_width_mm, tile_height_mm, cols, rows) <= 0:
            raise ValueError('Floor dimensions and tile counts must be positive')
        grid_count = (int(cols) + 1) * (int(rows) + 1)
        supplied_grid_indices = grid_indices is not None
        if (
            points.ndim != 2
            or points.shape[1:] != (2,)
            or len(points) < 4
            or (not supplied_grid_indices and len(points) not in (grid_count, 4, 6))
        ):
            raise ValueError(
                f'Provide at least four points from the {grid_count}-point floor grid'
            )
        uses_grid = supplied_grid_indices or len(points) == grid_count
        if uses_grid:
            if grid_indices is None:
                grid_indices = list(range(grid_count))
            grid_indices = np.asarray(grid_indices, dtype=int).reshape(-1)
            if len(grid_indices) != len(points):
                raise ValueError('Each selected image point needs a grid index')
            if len(np.unique(grid_indices)) != len(grid_indices):
                raise ValueError('Grid indices must be unique')
            if (grid_indices < 0).any() or (grid_indices >= grid_count).any():
                raise ValueError('Grid indices are outside the configured floor grid')
            if len(np.unique(points, axis=0)) != len(points):
                raise ValueError('Grid points must be unique')
        elif not is_ordered_boundary(points):
            raise ValueError('Floor points must be ordered around a non-degenerate boundary')
        w, h = image_size
        if (
            (points < 0).any()
            or (points[:, 0] >= w).any()
            or (points[:, 1] >= h).any()
        ):
            raise ValueError('Floor corners must lie inside the source image')
        world_w, world_h = cols * tile_width_mm, rows * tile_height_mm
        if uses_grid:
            world_points = np.array([
                [
                    (index % (cols + 1)) * tile_width_mm,
                    (index // (cols + 1)) * tile_height_mm,
                ]
                for index in grid_indices
            ], dtype=np.float32)
        elif len(points) == 6:
            world_points = np.array([
                [0, 0], [world_w, 0], [world_w, world_h / 2],
                [world_w, world_h], [0, world_h], [0, world_h / 2],
            ], dtype=np.float32)
        else:
            world_points = np.array([
                [0, 0], [world_w, 0], [world_w, world_h], [0, world_h],
            ], dtype=np.float32)
        matrix, _ = cv2.findHomography(
            points, world_points, 0,
        )
        if matrix is None:
            raise ValueError('Could not compute a floor coordinate transform')
        if uses_grid:
            coordinate_system = (
                'Grid points are row-major; origin is first point, +X toward '
                'the second point, +Y toward the first point of the second row; '
                'Z=0; units mm'
            )
        else:
            x_reference = 'second'
            y_reference = 'fifth' if len(points) == 6 else 'fourth'
            coordinate_system = (
                f'Origin first floor reference; +X toward {x_reference} point, '
                f'+Y toward {y_reference} point; Z=0; units mm'
            )
        metadata = {
            'schema_version': 1,
            'image_size': [int(w), int(h)],
            'image_to_world_mm': matrix.tolist(),
            'reference_pixels': points.tolist(),
            'tile_width_mm': tile_width_mm,
            'tile_height_mm': tile_height_mm,
            'tile_count': [cols, rows],
            'dimensions_measured': bool(measured),
            'coordinate_system': coordinate_system,
        }
        if uses_grid:
            metadata['reference_grid_indices'] = grid_indices.tolist()
        return cls(metadata)

    @classmethod
    def load(cls, path):
        """Load ``calibration.json`` produced by the calibration application."""
        # A relative path is resolved from the process working directory.
        # Use ``Path(__file__).resolve().parent / 'calibration.json'`` when
        # the file should follow this module across working directories.
        # FloorCalibration.load(Path(__file__).resolve().parent / 'calibration.json')
        return cls(json.loads(Path(path).read_text(encoding='utf-8')))

    def save(self, path):
        """Save calibration metadata without overwriting an existing file."""
        with Path(path).open('x', encoding='utf-8') as output:
            json.dump(self.metadata, output, ensure_ascii=False, indent=2)

    def validate_image_size(self, image_size):
        if tuple(image_size) != self.image_size:
            raise ValueError(
                'Image size changed; recalibrate after crop, resize, or camera changes'
            )

    def pixel_to_world(self, u, v):
        """Convert an image point to ``(x_mm, y_mm)`` on the floor plane."""
        return tuple(float(value) for value in transform_pixels([[u, v]], self.matrix)[0])

    def bbox_to_world(self, xyxy):
        """Approximate floor position from a detection-box center."""
        box = np.asarray(xyxy, dtype=np.float64)
        if (
            box.shape != (4,)
            or not np.isfinite(box).all()
            or box[2] <= box[0]
            or box[3] <= box[1]
        ):
            raise ValueError('Detection box must be a valid [x1, y1, x2, y2]')
        u, v = (box[:2] + box[2:]) / 2
        x, y = self.pixel_to_world(u, v)
        return {
            'image_uv': [float(u), float(v)],
            'floor_xy_mm': [x, y],
            'inside_calibrated_area': (
                0 <= x <= self.width_mm and 0 <= y <= self.height_mm
            ),
        }
