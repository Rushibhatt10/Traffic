"""Perspective transformation helpers for camera-based distance estimation."""

import cv2
import numpy as np


class PerspectiveTransformer:
    """Simple perspective transformer for bird-eye-style mapping."""

    def __init__(self, source_points, target_points):
        self.source_points = np.array(source_points, dtype=np.float32)
        self.target_points = np.array(target_points, dtype=np.float32)
        self.matrix = cv2.getPerspectiveTransform(self.source_points, self.target_points)

    def transform_point(self, point):
        point_array = np.array([[point[0], point[1]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point_array.reshape(-1, 1, 2), self.matrix)
        return transformed[0][0]
