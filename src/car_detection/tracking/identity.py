"""Resolve fixed vehicle identities from color evidence inside each box.

``car1`` is the yellow/blue car and ``car2`` is the red/blue car. Tracking
IDs are intentionally kept separate from these stable vehicle IDs.
"""

from collections import deque

import cv2
import numpy as np

from .motion import missing_record

VEHICLES = {1: 'yellow_blue', 2: 'red_blue'}


def classify_car(frame, box):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = np.rint(box).astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 - x1 < 5 or y2 - y1 < 5:
        return None, 0.0, {}
    hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    colorful = (saturation >= 80) & (value >= 45)
    yellow = colorful & (hue >= 16) & (hue <= 40)
    red = colorful & ((hue <= 15) | (hue >= 165))
    blue = colorful & (hue >= 85) & (hue <= 135)
    counts = np.array([yellow.sum(), red.sum(), blue.sum()], float)
    ratios = counts / hue.size
    features = {
        'yellow_ratio': float(ratios[0]),
        'red_ratio': float(ratios[1]),
        'blue_ratio': float(ratios[2]),
    }
    # Blue is shared by both cars and helps reject red cones as red/blue cars.
    if counts[2] < 5 or ratios[2] < 0.015:
        return None, 0.0, features
    dominance = counts[:2] / max(counts[:2].sum(), 1)
    winner = int(np.argmax(counts[:2]))
    if counts[winner] < 6 or ratios[winner] < 0.025 or dominance[winner] < 0.65:
        return None, 0.0, features
    confidence = float(dominance[winner] * min(1, ratios[winner] / 0.12))
    return winner + 1, confidence, features


class VehicleIdentityResolver:
    def __init__(self):
        self.votes = {}
        self.stable_identities = {}

    def resolve(self, frame, cars):
        visible = [car for car in cars if car.get('missed', 0) == 0]
        candidates = {1: [], 2: []}
        assigned = set()
        for car in visible:
            car['car_id'] = None
            track_id = car['track_id']
            identity, confidence, features = classify_car(frame, car['box'])
            votes = self.votes.setdefault(track_id, deque(maxlen=3))
            # An isolated low-confidence color sample should not erase a
            # previously stable identity.
            if identity is not None:
                votes.append(identity)
            car['identity_features'] = features
            car['identity_confidence'] = confidence
            stable = self.stable_identities.get(track_id)
            if stable in VEHICLES and stable not in assigned:
                car['car_id'] = stable
                car['vehicle_name'] = VEHICLES[stable]
                assigned.add(stable)
                continue
            # Require two consistent samples, or one very strong sample, for
            # a new track. Once assigned, the identity is held through weak
            # color frames and short detector gaps.
            if (
                identity is not None
                and (
                    sum(v == identity for v in votes) >= 2
                    or confidence >= 0.8
                )
            ):
                candidates[identity].append(car)
        for identity, items in candidates.items():
            if identity in assigned:
                continue
            items.sort(
                key=lambda car: car['identity_confidence'] * car['score'],
                reverse=True,
            )
            if not items:
                continue
            # Treat similarly confident candidates for one identity as a
            # conflict instead of forcing an assignment.
            if len(items) > 1:
                first = items[0]['identity_confidence'] * items[0]['score']
                second = items[1]['identity_confidence'] * items[1]['score']
                if first - second < 0.15:
                    continue
            items[0]['car_id'] = identity
            items[0]['vehicle_name'] = VEHICLES[identity]
            track_id = items[0]['track_id']
            self.stable_identities[track_id] = identity
            assigned.add(identity)

        # The two-scene setup contains exactly two vehicles. If one vehicle
        # is confidently identified and the other visible track has weak or
        # missing color evidence, assign the only remaining identity instead
        # of displaying ``unknown``. Do not infer with a single visible car.
        if len(visible) == 2 and len(assigned) == 1:
            unresolved = [car for car in visible if car.get('car_id') is None]
            remaining = [identity for identity in VEHICLES if identity not in assigned]
            if len(unresolved) == 1 and len(remaining) == 1:
                car = unresolved[0]
                identity = remaining[0]
                car['car_id'] = identity
                car['vehicle_name'] = VEHICLES[identity]
                car['identity_inferred'] = True
                car['identity_confidence'] = max(
                    float(car.get('identity_confidence', 0.0)), 0.35,
                )
                self.stable_identities[car['track_id']] = identity
        return visible


def vehicle_missing_record(car_id, timestamp_us):
    record = missing_record(car_id, timestamp_us)
    record['track_id'] = None
    record['vehicle_name'] = VEHICLES[car_id]
    return record
