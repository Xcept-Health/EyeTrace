"""
Utility functions for head_pose module.
"""

import numpy as np
import cv2

def get_face_landmark_array(face_landmarks, indices, width, height):
    """
    Extract specific landmarks as array of pixel coordinates.
    """
    points = []
    for idx in indices:
        lm = face_landmarks.landmark[idx]
        x = int(lm.x * width)
        y = int(lm.y * height)
        points.append([x, y])
    return np.array(points, dtype=np.float64)


# 3D model points of a generic face (in mm) for solvePnP
# These correspond to MediaPipe indices: chin, left eye left, right eye right, mouth left, mouth right, nose tip, etc.
# A common set: (nose tip, chin, left eye left, right eye right, left mouth, right mouth)
# MediaPipe indices: 1 (nose tip), 152 (chin), 33 (left eye left), 263 (right eye right), 61 (left mouth), 291 (right mouth)
FACE_MODEL_POINTS = np.array([
    [0.0, 0.0, 0.0],        # nose tip
    [0.0, -330.0, -65.0],   # chin
    [-225.0, 170.0, -135.0], # left eye left
    [225.0, 170.0, -135.0],  # right eye right
    [-150.0, -150.0, -125.0], # left mouth
    [150.0, -150.0, -125.0]   # right mouth
], dtype=np.float64)

FACE_MODEL_INDICES = [1, 152, 33, 263, 61, 291]