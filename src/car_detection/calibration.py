"""Interactive camera snapshot and floor-grid calibration selection."""

import cv2
import numpy as np


def snapshot(cap):
    """Wait for the user to capture one calibration frame."""
    window = "Snapshot | S: freeze calibration frame | Q/ESC: cancel"
    cv2.namedWindow(window)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(
                    "No camera/video frame. Check source and camera permissions."
                )
            scale = min(1, 950 / frame.shape[1], 750 / frame.shape[0])
            shown = cv2.resize(frame, None, fx=scale, fy=scale)
            cv2.putText(
                shown,
                "S: capture | Q: cancel",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.imshow(window, shown)
            key = cv2.waitKey(30) & 255
            if key == ord("s"):
                return frame
            if key in (27, ord("q")) or cv2.getWindowProperty(
                window, cv2.WND_PROP_VISIBLE
            ) < 1:
                return None
    finally:
        cv2.destroyAllWindows()


def choose_corners(frame, cols: int, rows: int):
    """Collect ordered floor-grid points and return points plus grid indices."""
    scale = min(1, 950 / frame.shape[1], 750 / frame.shape[0])
    preview = cv2.resize(frame, None, fx=scale, fy=scale)
    points = []
    grid_cols = cols + 1
    grid_rows = rows + 1
    expected_points = grid_cols * grid_rows
    window = (
        f"Select {expected_points} floor grid points | "
        "ENTER accept | R reset | ESC cancel"
    )
    cv2.namedWindow(window)

    def click(event, x, y, flags, param):
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < expected_points:
            points.append([x / scale, y / scale])

    cv2.setMouseCallback(window, click)
    try:
        while True:
            canvas = preview.copy()
            for index, point in enumerate(points):
                if point is None:
                    continue
                location = tuple(np.rint(np.array(point) * scale).astype(int))
                cv2.circle(canvas, location, 5, (0, 255, 0), -1)
                row, col = divmod(index, grid_cols)
                cv2.putText(
                    canvas,
                    f"R{row + 1}C{col + 1}",
                    (location[0] + 8, location[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    2,
                )

            for row in range(grid_rows):
                segment = []
                for col in range(grid_cols):
                    index = row * grid_cols + col
                    point = points[index] if index < len(points) else None
                    if point is None:
                        if len(segment) > 1:
                            row_points = np.rint(
                                np.array(segment) * scale
                            ).astype(int)
                            cv2.polylines(
                                canvas, [row_points], False, (0, 255, 0), 1
                            )
                        segment = []
                    else:
                        segment.append(point)
                if len(segment) > 1:
                    row_points = np.rint(np.array(segment) * scale).astype(int)
                    cv2.polylines(canvas, [row_points], False, (0, 255, 0), 1)

            for col in range(grid_cols):
                segment = []
                for row in range(grid_rows):
                    index = row * grid_cols + col
                    point = points[index] if index < len(points) else None
                    if point is None:
                        if len(segment) > 1:
                            column_points = np.rint(
                                np.array(segment) * scale
                            ).astype(int)
                            cv2.polylines(
                                canvas, [column_points], False, (0, 255, 0), 1
                            )
                        segment = []
                    else:
                        segment.append(point)
                if len(segment) > 1:
                    column_points = np.rint(np.array(segment) * scale).astype(int)
                    cv2.polylines(canvas, [column_points], False, (0, 255, 0), 1)

            if len(points) < expected_points:
                next_row, next_col = divmod(len(points), grid_cols)
                message = (
                    f"Click row {next_row + 1}, column {next_col + 1} "
                    "| C: skip | ENTER: accept after 4+ points"
                )
            else:
                message = "ENTER: accept | R: reset"
            cv2.putText(
                canvas,
                message,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
            cv2.imshow(window, canvas)
            key = cv2.waitKey(20) & 255
            if key in (27, ord("q")) or cv2.getWindowProperty(
                window, cv2.WND_PROP_VISIBLE
            ) < 1:
                return None
            if key == ord("r"):
                points.clear()
            if key in (ord("c"), ord("C")) and len(points) < expected_points:
                points.append(None)
                continue
            if key in (10, 13):
                valid_indices = [
                    index for index, point in enumerate(points) if point is not None
                ]
                if len(valid_indices) >= 4:
                    valid_points = np.array(
                        [points[index] for index in valid_indices], np.float32
                    )
                    return valid_points, valid_indices
                print(
                    "Select or skip at least four points before accepting.",
                    flush=True,
                )
    finally:
        cv2.destroyAllWindows()
