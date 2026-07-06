"""Professional vehicle counting based on virtual line crossing."""

import cv2
import config


class VehicleCounter:
    """Count vehicles only when they cross the configured counting line."""

    def __init__(self):
        self.previous_positions = {}
        self.counted_ids = set()
        self.total_count = 0
        self.vehicle_count = {
            "Car": 0,
            "Motorcycle": 0,
            "Bus": 0,
            "Truck": 0,
        }

    def update_vehicle(self, track_id, vehicle_name, center):
        """Update tracking state and count a vehicle if the line is crossed."""
        previous = self.previous_positions.get(track_id)
        if previous is None:
            self.previous_positions[track_id] = center
            return

        if self._crossed_line(previous, center):
            if track_id not in self.counted_ids:
                self.counted_ids.add(track_id)
                self.total_count += 1
                self.vehicle_count[vehicle_name] += 1

        self.previous_positions[track_id] = center

    def _crossed_line(self, previous_center, current_center):
        if config.COUNTING_AXIS == "x":
            previous_x = previous_center[0]
            current_x = current_center[0]
            if config.COUNTING_DIRECTION == "right":
                return previous_x < config.LINE_X and current_x >= config.LINE_X
            if config.COUNTING_DIRECTION == "left":
                return previous_x > config.LINE_X and current_x <= config.LINE_X
            return False

        previous_y = previous_center[1]
        current_y = current_center[1]
        if config.COUNTING_DIRECTION == "down":
            return previous_y < config.LINE_Y and current_y >= config.LINE_Y
        if config.COUNTING_DIRECTION == "up":
            return previous_y > config.LINE_Y and current_y <= config.LINE_Y
        return False

    def draw(self, frame):
        """Draw the counting line and live summary statistics."""
        h, w = frame.shape[:2]
        if config.COUNTING_AXIS == "x":
            cv2.line(frame, (config.LINE_X, 0), (config.LINE_X, h), config.LINE_COLOR, 3)
        else:
            cv2.line(frame, (0, config.LINE_Y), (w, config.LINE_Y), config.LINE_COLOR, 3)
        cv2.putText(frame, f"Total : {self.total_count}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, config.TEXT_COLOR, 2)
        cv2.putText(frame, f"Cars : {self.vehicle_count['Car']}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Motorcycles : {self.vehicle_count['Motorcycle']}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Buses : {self.vehicle_count['Bus']}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Trucks : {self.vehicle_count['Truck']}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)