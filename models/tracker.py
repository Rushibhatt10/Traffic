"""Vehicle detection and tracking wrapper using Ultralytics YOLO."""

import math

from ultralytics import YOLO

import config


class VehicleTracker:
    """Wrapper around YOLO detection with a lightweight fallback tracker."""

    def __init__(self):
        self.model = YOLO(config.MODEL_PATH)
        self.previous_boxes = {}
        self.next_track_id = 1

    def track(self, frame):
        """Run object detection and tracking on a single frame."""
        try:
            results = self.model(
                frame,
                conf=config.CONFIDENCE,
                classes=config.VEHICLE_CLASSES,
                imgsz=320,
                stream=False,
                verbose=False,
            )
            return results
        except Exception:
            return []

    def get_track_ids(self, boxes):
        """Assign stable track IDs when the detector does not return them."""
        if boxes.id is not None:
            return [int(track_id) for track_id in boxes.id.tolist()]

        assigned_ids = []
        for box in boxes.xyxy.tolist():
            x1, y1, x2, y2 = box
            center = ((x1 + x2) / 2, (y1 + y2) / 2)

            best_id = None
            best_distance = float("inf")
            for prev_id, prev_data in self.previous_boxes.items():
                prev_center = prev_data["center"]
                distance = math.hypot(center[0] - prev_center[0], center[1] - prev_center[1])
                if distance < best_distance and distance < 180:
                    best_distance = distance
                    best_id = prev_id

            if best_id is None:
                best_id = self.next_track_id
                self.next_track_id += 1

            assigned_ids.append(best_id)
            self.previous_boxes[best_id] = {"center": center}

        if len(self.previous_boxes) > 200:
            self.previous_boxes = {track_id: data for track_id, data in list(self.previous_boxes.items())[-100:]}

        return assigned_ids