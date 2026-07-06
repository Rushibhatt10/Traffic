"""Wrong-side vehicle detection using trajectory analysis."""

from collections import defaultdict

import cv2
import numpy as np

import config


class WrongSideDetector:
    """Detect vehicles moving opposite to the configured direction."""

    def __init__(self):
        self.positions = defaultdict(list)
        self.wrong_side_events = []
        self.wrong_side_ids = set()
        self.last_direction = {}

    def update_vehicle(self, track_id, vehicle_name, center):
        history = self.positions[track_id]
        history.append(center)

        if len(history) < 2:
            return

        prev = history[-2]
        curr = history[-1]
        dx = curr[0] - prev[0]
        dy = curr[1] - prev[1]

        if abs(dx) + abs(dy) < config.MIN_MOVEMENT_PIXELS:
            return

        direction = "down" if dy > 0 else "up" if dy < 0 else "stationary"
        self.last_direction[track_id] = direction

        if direction != config.ALLOWED_DIRECTION and direction != "stationary":
            if len(history) >= config.WRONG_SIDE_THRESHOLD_FRAMES:
                if track_id not in self.wrong_side_ids:
                    self.wrong_side_ids.add(track_id)
                    self.wrong_side_events.append({
                        "track_id": track_id,
                        "vehicle_type": vehicle_name,
                        "direction": direction,
                    })

    def draw(self, frame):
        for track_id, history in self.positions.items():
            if len(history) > 1:
                pts = [(x, y) for x, y in history[-20:]]
                if pts:
                    cv2.polylines(frame, [np.array(pts, dtype=np.int32)], False, config.WRONG_SIDE_COLOR, 1)
            if track_id in self.wrong_side_ids:
                last = history[-1]
                cv2.putText(frame, "WRONG SIDE", (last[0], last[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.WRONG_SIDE_COLOR, 2)
