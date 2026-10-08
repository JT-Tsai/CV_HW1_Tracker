"""Fast color/saliency ROI selection shared by labeling and inference."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage


@dataclass(frozen=True)
class Region:
    """An image region in pixel coordinates, using xyxy convention."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.x0, self.y0, self.x1, self.y1


def _clip_region(region: Region, width: int, height: int) -> Region:
    return Region(
        max(0, min(region.x0, width)),
        max(0, min(region.y0, height)),
        max(0, min(region.x1, width)),
        max(0, min(region.y1, height)),
    )


def _merge_regions(regions: list[Region]) -> list[Region]:
    """Merge overlapping regions so one object is not cropped repeatedly."""
    merged = list(regions)
    changed = True
    while changed:
        changed = False
        output: list[Region] = []
        while merged:
            current = merged.pop(0)
            index = 0
            while index < len(merged):
                other = merged[index]
                overlaps = not (
                    current.x1 < other.x0
                    or other.x1 < current.x0
                    or current.y1 < other.y0
                    or other.y1 < current.y0
                )
                if overlaps:
                    current = Region(
                        min(current.x0, other.x0),
                        min(current.y0, other.y0),
                        max(current.x1, other.x1),
                        max(current.y1, other.y1),
                    )
                    merged.pop(index)
                    changed = True
                else:
                    index += 1
            output.append(current)
        merged = output
    return merged


def select_regions(
    image: Image.Image,
    *,
    padding: int = 256,
    min_area: int = 20,
    dilation: int = 12,
    merge: bool = True,
) -> tuple[np.ndarray, list[Region]]:
    """Select red/high-saturation components and return a debug mask and ROIs."""
    image_array = np.asarray(image.convert("RGB")).astype(np.int16)
    return _select_regions_from_channels(
        image_array[..., 0], image_array[..., 1], image_array[..., 2],
        padding=padding, min_area=min_area, dilation=dilation, merge=merge,
    )


def select_regions_bgr(
    image: np.ndarray,
    *,
    padding: int = 256,
    min_area: int = 20,
    dilation: int = 12,
    merge: bool = True,
) -> tuple[np.ndarray, list[Region]]:
    """Select color components directly from an OpenCV BGR frame."""
    image_array = np.asarray(image)
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError('BGR image must have shape (height, width, 3)')
    image_array = image_array.astype(np.int16, copy=False)
    return _select_regions_from_channels(
        image_array[..., 2], image_array[..., 1], image_array[..., 0],
        padding=padding, min_area=min_area, dilation=dilation, merge=merge,
    )


def _select_regions_from_channels(
    red: np.ndarray,
    green: np.ndarray,
    blue: np.ndarray,
    *,
    padding: int,
    min_area: int,
    dilation: int,
    merge: bool,
) -> tuple[np.ndarray, list[Region]]:
    red_mask = (
        (red > 90)
        & (red > green + 25)
        & (red > blue + 25)
        & (red > green * 1.25)
    )
    maximum = np.maximum(np.maximum(red, green), blue)
    minimum = np.minimum(np.minimum(red, green), blue)
    color_range = maximum - minimum
    colorful_mask = (color_range > 35) & (maximum > 70)
    filter_mask = red_mask | colorful_mask

    # Match SciPy's cross-shaped neighborhood and zero-padding behavior.
    # For dilation <= 0, preserve SciPy's special repeated-dilation semantics.
    if dilation > 0:
        cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
        grouped_mask = cv2.dilate(
            filter_mask.astype(np.uint8),
            cross,
            iterations=dilation,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(bool)
    else:
        grouped_mask = ndimage.binary_dilation(filter_mask, iterations=dilation)
    labels, component_count = ndimage.label(grouped_mask)
    # Count source-color pixels once instead of rescanning the full image per
    # component. This matches filter_mask[labels == component_id].sum().
    component_counts = np.bincount(labels[filter_mask], minlength=component_count + 1)
    height, width = filter_mask.shape
    regions: list[Region] = []
    for component_id, component_slice in enumerate(
        ndimage.find_objects(labels), start=1
    ):
        if component_slice is None:
            continue
        component_pixels = int(component_counts[component_id])
        if component_pixels < min_area:
            continue
        y_slice, x_slice = component_slice
        regions.append(
            _clip_region(
                Region(
                    x_slice.start - padding,
                    y_slice.start - padding,
                    x_slice.stop + padding,
                    y_slice.stop + padding,
                ),
                width,
                height,
            )
        )
    return filter_mask, _merge_regions(regions) if merge else regions


def square_region(region: Region, width: int, height: int, minimum: int = 512) -> Region:
    """Make a padded region square while keeping it inside the image."""
    side = min(max(region.width, region.height, minimum), max(width, height))
    center_x = (region.x0 + region.x1) // 2
    center_y = (region.y0 + region.y1) // 2
    x0 = max(0, min(center_x - side // 2, width - side))
    y0 = max(0, min(center_y - side // 2, height - side))
    return Region(x0, y0, min(width, x0 + side), min(height, y0 + side))


def tile_region(
    region: Region,
    width: int,
    height: int,
    *,
    crop_size: int = 1024,
    overlap: float = 0.2,
    minimum_square: int = 512,
) -> list[Region]:
    """Split a region into square crops when it exceeds the crop size."""
    region = _clip_region(region, width, height)
    if region.width <= crop_size and region.height <= crop_size:
        return [square_region(region, width, height, minimum_square)]

    stride = max(1, int(round(crop_size * (1.0 - overlap))))
    xs = list(range(region.x0, max(region.x1 - crop_size, region.x0) + 1, stride))
    ys = list(range(region.y0, max(region.y1 - crop_size, region.y0) + 1, stride))
    last_x = max(region.x1 - crop_size, region.x0)
    last_y = max(region.y1 - crop_size, region.y0)
    if not xs or xs[-1] != last_x:
        xs.append(last_x)
    if not ys or ys[-1] != last_y:
        ys.append(last_y)
    return [
        Region(x, y, min(x + crop_size, width), min(y + crop_size, height))
        for y in ys
        for x in xs
    ]
