"""Estimate millimeter positions and motion on the calibrated floor plane.

Angles start at +X; +Y points down, so positive angles rotate clockwise.
"""
from collections import deque

import numpy as np


class MotionEstimator:
    def __init__(
        self,
        window_seconds=0.25,
        min_span_seconds=0.1,
        min_speed_mm_s=30,
        min_displacement_mm=8,
    ):
        self.window = window_seconds
        self.min_span = min_span_seconds
        self.history = {}
        self.min_speed = min_speed_mm_s
        self.min_displacement = min_displacement_mm
        self.last_movement_direction = {}

    def update(self, car_id, timestamp_us, position):
        """Add measured frames and use a short time window to reduce jitter."""
        timestamp = timestamp_us / 1e6
        history = self.history.setdefault(car_id, deque(maxlen=120))
        if history and timestamp <= history[-1][0]:
            raise ValueError('Timestamps must be strictly increasing')
        position = np.asarray(position, float)
        if history:
            dt = timestamp - history[-1][0]
            # Restart after a long gap or implausibly large jump.
            if dt > 0.5 or np.linalg.norm(position - history[-1][1]) > 50 + 2000 * dt:
                history.clear()
        history.append((timestamp, position))
        while len(history) > 1 and timestamp - history[0][0] > self.window:
            history.popleft()
        velocity = None
        if len(history) >= 2 and timestamp-history[0][0] >= self.min_span:
            times = np.array([row[0] for row in history])
            positions = np.array([row[1] for row in history])
            # Use the median of pairwise velocities to reduce box-center jitter.
            slopes=[]
            for i in range(len(times) - 1):
                for j in range(i + 1, len(times)):
                    dt = times[j] - times[i]
                    if dt >= self.min_span - 1e-9:
                        slopes.append((positions[j] - positions[i]) / dt)
            if slopes:
                velocity = np.median(slopes, axis=0).tolist()
        speed = None if velocity is None else float(np.linalg.norm(velocity))
        # Travel direction is based on the trajectory, not body heading.
        movement = None
        movement_vector = None
        # Use only recent net displacement for the arrow. This avoids a
        # direction flip when one detector box jitters inside the short window.
        recent = [row for row in history if timestamp - row[0] <= 0.12 + 1e-9]
        if len(recent) < 2 and len(history) >= 2:
            recent = list(history)[-2:]  # Keep two valid samples at low frame rates.
        if len(recent) >= 2:
            movement_vector = recent[-1][1] - recent[0][1]
            displacement = float(np.linalg.norm(movement_vector))
        else:
            displacement = 0.0
        direction_vector = movement_vector
        if (
            direction_vector is None
            or displacement < self.min_displacement
        ) and velocity is not None:
            # Keep the arrow visible when the short net displacement is small
            # because of perspective or detector jitter.
            direction_vector = np.asarray(velocity, dtype=float)
        if (
            speed is not None
            and speed >= self.min_speed
            and direction_vector is not None
            and np.linalg.norm(direction_vector) > 1e-9
        ):
            raw_movement = float(
                np.degrees(np.arctan2(direction_vector[1], direction_vector[0])) % 360
            )
            previous = self.last_movement_direction.get(car_id)
            if previous is None:
                movement = raw_movement
            else:
                # Smooth on a circle so 359 -> 1 degrees does not jump
                # through 180 degrees. Limit each update to reduce jitter.
                delta = (raw_movement - previous + 180.0) % 360.0 - 180.0
                delta = float(np.clip(delta, -45.0, 45.0))
                movement = (previous + 0.35 * delta) % 360.0
            self.last_movement_direction[car_id] = movement
        return {
            'velocity_xy_mm_s': velocity,
            'speed_mm_s': speed,
            'movement_direction_deg': movement,
        }

    def mark_missing(self, car_id):
        # Restart velocity accumulation when the vehicle reappears.
        self.history.pop(car_id, None)
        self.last_movement_direction.pop(car_id, None)


def missing_record(car_id, timestamp_us):
    return {
        'car_id': car_id,
        'track_id': car_id,
        'timestamp_us': timestamp_us,
        'visible': False,
        'floor_xy_mm': [-1000.0, -1000.0],
        'image_uv': None,
        'velocity_xy_mm_s': None,
        'speed_mm_s': None,
        'movement_direction_deg': None,
    }
