"""Calibration persistence for road geometry settings."""

import json
from pathlib import Path

import config


class CalibrationManager:
    """Load and save calibration points and parameters."""

    def __init__(self, file_path=None):
        self.file_path = Path(file_path or config.CALIBRATION_FILE)
        self.points = list(config.CALIBRATION_POINTS)

    def save(self):
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        with self.file_path.open("w", encoding="utf-8") as handle:
            json.dump({"points": self.points}, handle, indent=2)

    def load(self):
        if self.file_path.exists():
            with self.file_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            self.points = data.get("points", self.points)
        return self.points
