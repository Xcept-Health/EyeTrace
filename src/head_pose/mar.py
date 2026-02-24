"""
Mouth Aspect Ratio (MAR) for yawn detection.
"""

import numpy as np

# MediaPipe indices for mouth: outer points (typically 61, 291, 0, 17, 267, 314, etc.)
# For a simple MAR, we use: top lip, bottom lip, left corner, right corner.
# Let's define indices for outer mouth: [61, 291, 0, 17, 267, 314]? Need standard set.
# Common set for MAR: (61, 291, 0, 17, 267, 314) but we need 6 points like EAR.
# Simpler: use vertical and horizontal distances.

MOUTH_TOP_IDX = 13      # upper lip top? Actually MediaPipe has many. We'll use common landmarks.
MOUTH_BOTTOM_IDX = 14
MOUTH_LEFT_IDX = 61
MOUTH_RIGHT_IDX = 291

def mouth_aspect_ratio(face_landmarks, image_width, image_height):
    """
    Compute Mouth Aspect Ratio (MAR) similar to EAR but for mouth.

    MAR = (vertical distance) / (horizontal distance)
    Vertical: distance between top and bottom inner lip.
    Horizontal: distance between left and right mouth corners.

    Parameters
    ----------
    face_landmarks : mediapipe face landmarks
    image_width, image_height : int

    Returns
    -------
    float
        MAR value.
    """
    def get_point(idx):
        lm = face_landmarks.landmark[idx]
        return np.array([lm.x * image_width, lm.y * image_height])

    top = get_point(13)   # adjust indices as needed
    bottom = get_point(14)
    left = get_point(61)
    right = get_point(291)

    vertical = np.linalg.norm(top - bottom)
    horizontal = np.linalg.norm(left - right)

    if horizontal == 0:
        return 0.0
    return vertical / horizontal