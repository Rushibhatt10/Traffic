"""Camera-based speed estimation using center-point motion and frame timestamps."""

from collections import defaultdict

import cv2
import numpy as np

import config


class SpeedEstimator:
    """Estimate vehicle speed in km/h from pixel movement."""

    def __init__(self, transform=None, transformer=None):
        self.positions = defaultdict(list)
        self.speeds = defaultdict(list)
        self.overspeed_events = []
        self.transformer = transformer
        self.transform = transform

    def update_vehicle(self, track_id, vehicle_name, center, frame_idx):
        history = self.positions[track_id]
        history.append((center, frame_idx))

        if len(history) < 2:
            return

        prev_center, prev_frame = history[-2]
        curr_center, curr_frame = history[-1]

        distance_pixels = np.linalg.norm(np.array(curr_center) - np.array(prev_center))
        if distance_pixels < 1:
            return

        elapsed_seconds = max(1e-6, (curr_frame - prev_frame) / config.FPS)
        distance_meters = distance_pixels * config.PIXEL_TO_METER
        speed_kmh = (distance_meters / elapsed_seconds) * 3.6

        if speed_kmh > config.MAX_REASONABLE_SPEED_KMH:
            speed_kmh = config.MAX_REASONABLE_SPEED_KMH

        self.speeds[track_id].append(speed_kmh)
        if len(self.speeds[track_id]) > config.SPEED_SMOOTHING_WINDOW:
            self.speeds[track_id] = self.speeds[track_id][-config.SPEED_SMOOTHING_WINDOW:]

        smoothed = float(np.mean(self.speeds[track_id]))
        if smoothed > config.SPEED_LIMIT:
            if track_id not in {event["track_id"] for event in self.overspeed_events}:
                self.overspeed_events.append({
                    "track_id": track_id,
                    "vehicle_type": vehicle_name,
                    "speed": smoothed,
                })

    def draw(self, frame):
        for track_id, history in self.positions.items():
            if len(history) < 2:
                continue
            latest_center, _ = history[-1]
            speed_values = self.speeds.get(track_id, [])
            if speed_values:
                speed_kmh = float(np.mean(speed_values[-3:]))
                label = f"{speed_kmh:.1f} km/h"
                color = config.OVERSPEED_COLOR if speed_kmh > config.SPEED_LIMIT else config.TEXT_COLOR
                cv2.putText(frame, label, (latest_center[0], latest_center[1] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
