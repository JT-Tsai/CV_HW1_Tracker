"""Camera and top-down floor visualization."""

import cv2
import numpy as np

from .geometry.floor import transform_pixels
from .tracking.identity import vehicle_missing_record


def map_point(point, matrix):
    return transform_pixels([point], matrix)[0]


def vehicle_ground_pixel(box):
    """Return the image point nearest the vehicle's ground contact point."""
    x1, y1, x2, y2 = np.asarray(box, dtype=float)
    return [(x1 + x2) * 0.5, y2]


def draw(
    frame,
    cars,
    matrix,
    world_width,
    world_height,
    measured,
    cols=2,
    rows=3,
    states=None,
):
    """Render the live frame, floor projection, tracks, and motion state."""
    canvas = np.full((800, 1200, 3), 25, np.uint8)
    ratio = min(680 / frame.shape[1], 600 / frame.shape[0])
    shown = cv2.resize(frame, None, fx=ratio, fy=ratio)
    coordinates = []
    map_scale = min(430 / world_width, 620 / world_height)
    map_width = max(2, round(world_width * map_scale))
    map_height = max(2, round(world_height * map_scale))
    map_x, map_y = 730, 100
    pixel_scale = np.diag(
        [(map_width - 1) / world_width, (map_height - 1) / world_height, 1.0]
    )
    bird_view = cv2.warpPerspective(
        frame, pixel_scale @ matrix, (map_width, map_height)
    )
    inverse = np.linalg.inv(matrix)
    grid_lines = []
    for col in range(cols + 1):
        x = world_width * col / cols
        grid_lines.append([[x, 0], [x, world_height]])
    for row in range(rows + 1):
        y = world_height * row / rows
        grid_lines.append([[0, y], [world_width, y]])

    for endpoints in grid_lines:
        source_line = transform_pixels(endpoints, inverse) * ratio
        if np.isfinite(source_line).all() and np.abs(source_line).max() < 1e8:
            start, end = np.rint(source_line).astype(int)
            visible, start, end = cv2.clipLine(
                (0, 0, shown.shape[1], shown.shape[0]),
                tuple(start),
                tuple(end),
            )
            if visible:
                cv2.line(shown, start, end, (0, 255, 255), 1, cv2.LINE_AA)
        target_line = transform_pixels(endpoints, pixel_scale)
        start, end = np.rint(target_line).astype(int)
        cv2.line(
            bird_view,
            tuple(start),
            tuple(end),
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    canvas[map_y : map_y + map_height, map_x : map_x + map_width] = bird_view
    cv2.rectangle(
        canvas,
        (map_x, map_y),
        (map_x + map_width - 1, map_y + map_height - 1),
        (0, 255, 255),
        2,
    )

    for car in cars:
        if car.get("missed", 0) > 0:
            continue
        x1, y1, x2, y2 = car["box"]
        image_uv = vehicle_ground_pixel(car["box"])
        floor_xy = map_point(image_uv, matrix)
        inside = bool(
            0 <= floor_xy[0] <= world_width and 0 <= floor_xy[1] <= world_height
        )
        car_id = car.get("car_id")
        label_id = (
            f"car{car_id}" if car_id is not None else f'unknown (track {car["track_id"]})'
        )
        coordinates.append(
            {
                "car_id": car_id,
                "track_id": car["track_id"],
                "image_uv": image_uv,
                "floor_xy_mm": floor_xy.tolist(),
                "inside_calibrated_area": inside,
                "score": car["score"],
            }
        )
        start = tuple(np.rint(np.array([x1, y1]) * ratio).astype(int))
        end = tuple(np.rint(np.array([x2, y2]) * ratio).astype(int))
        cv2.rectangle(shown, start, end, (0, 255, 0), 2)
        cv2.putText(
            shown,
            f"{label_id}: {floor_xy[0]:.0f}, {floor_xy[1]:.0f} mm",
            (max(0, start[0] - 20), max(18, start[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (0, 255, 0),
            1,
        )
        state = (states or {}).get(car_id, {}) if car_id is not None else {}
        movement = state.get("movement_direction_deg")
        if movement is not None and state.get("visible", True):
            radians = np.radians(movement)
            end_xy = floor_xy + 80 * np.array([np.cos(radians), np.sin(radians)])
            image_arrow = transform_pixels([floor_xy, end_xy], inverse) * ratio
            if np.isfinite(image_arrow).all() and np.abs(image_arrow).max() < 1e8:
                arrow_start, arrow_end = np.rint(image_arrow).astype(int)
                visible, arrow_start, arrow_end = cv2.clipLine(
                    (0, 0, shown.shape[1], shown.shape[0]),
                    tuple(arrow_start),
                    tuple(arrow_end),
                )
                if visible:
                    cv2.arrowedLine(
                        shown,
                        arrow_start,
                        arrow_end,
                        (255, 255, 0),
                        2,
                        tipLength=0.3,
                    )
        if inside:
            box_pixels = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            floor_box = transform_pixels(box_pixels, pixel_scale @ matrix)
            if np.isfinite(floor_box).all() and np.abs(floor_box).max() < 1e8:
                for corner in range(4):
                    start, end = np.rint(
                        floor_box[[corner, (corner + 1) % 4]]
                    ).astype(int)
                    ok_line, start, end = cv2.clipLine(
                        (0, 0, map_width, map_height), tuple(start), tuple(end)
                    )
                    if ok_line:
                        cv2.line(
                            canvas,
                            (map_x + start[0], map_y + start[1]),
                            (map_x + end[0], map_y + end[1]),
                            (0, 255, 0),
                            2,
                        )
            location = (
                map_x + round(floor_xy[0] * (map_width - 1) / world_width),
                map_y + round(floor_xy[1] * (map_height - 1) / world_height),
            )
            label = f"{label_id} ({floor_xy[0]:.0f},{floor_xy[1]:.0f}) mm"
            text_y = min(map_y + map_height - 5, max(map_y + 18, location[1] - 10))
            cv2.putText(
                canvas, label, (map_x + 4, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (0, 0, 0), 3,
            )
            cv2.putText(
                canvas, label, (map_x + 4, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (0, 255, 0), 1,
            )
            if movement is not None:
                radians = np.radians(movement)
                tip = (
                    location[0] + round(30 * np.cos(radians)),
                    location[1] + round(30 * np.sin(radians)),
                )
                cv2.arrowedLine(canvas, location, tip, (255, 255, 0), 2, tipLength=0.3)

    canvas[75 : 75 + shown.shape[0], : shown.shape[1]] = shown
    cv2.putText(
        canvas, "CAR DETECTION + GROUND COORDINATES", (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
    )
    cv2.putText(
        canvas,
        "MEASURED TILE DIMENSIONS" if measured else "DEMO TILE DIMENSIONS - NOT MEASURED",
        (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
    )
    cv2.putText(
        canvas, "LIVE FLOOR VIEW | X right, Y down", (730, 75),
        cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1,
    )
    cv2.putText(
        canvas, f"Visible car tracks: {len(coordinates)}", (12, 740),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
    )
    cv2.putText(
        canvas, "CYAN ARROW: MOVEMENT, NOT BODY HEADING", (12, 680),
        cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 0), 1,
    )
    if not coordinates:
        cv2.putText(
            canvas, "NO CAR: (-1000.0, -1000.0)",
            (map_x + 6, map_y + map_height + 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 180, 255), 1,
        )

    display_states = [
        (states or {}).get(car_id, vehicle_missing_record(car_id, 0))
        for car_id in (1, 2)
    ]
    for row, state in enumerate(display_states):
        if not state.get("visible", True):
            cv2.putText(
                canvas,
                f'car{state["car_id"]}: (-1000.0, -1000.0) mm - NOT IDENTIFIED',
                (12, 700 + row * 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (0, 180, 255),
                1,
            )
            continue
        theta = state.get("movement_direction_deg")
        velocity = state.get("velocity_xy_mm_s")
        speed = state.get("speed_mm_s")
        theta_text = "N/A" if theta is None else f"{theta:.1f}"
        velocity_text = "N/A" if velocity is None else f"({velocity[0]:.0f},{velocity[1]:.0f})"
        speed_text = "N/A" if speed is None else f"{speed:.0f}"
        x, y = state["floor_xy_mm"]
        text = (
            f'car{state["car_id"]}: ({x:.1f},{y:.1f}) mm | '
            f"move {theta_text} | v {velocity_text} | speed {speed_text}"
        )
        cv2.putText(
            canvas, text, (12, 700 + row * 18), cv2.FONT_HERSHEY_SIMPLEX,
            0.39, (255, 255, 255), 1,
        )
    cv2.putText(
        canvas,
        "Q: quit | Fixed camera required | Box center is a ground-position proxy",
        (12, 775), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1,
    )
    return canvas, coordinates
