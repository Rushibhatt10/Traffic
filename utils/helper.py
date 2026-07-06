"""Utility helpers for the Traffic Survey AI project."""

from pathlib import Path


def ensure_directory(path):
    """Create a directory if it does not already exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def vehicle_name_from_class_id(class_id):
    """Map a COCO class ID to the supported vehicle name."""
    mapping = {
        2: "Car",
        3: "Motorcycle",
        5: "Bus",
        7: "Truck",
    }
    return mapping.get(int(class_id))
