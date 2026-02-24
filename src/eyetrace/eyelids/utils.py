"""
Utility functions for eyelid module, mainly landmark extraction.
"""

import numpy as np

# MediaPipe indices for the 6 key points of each eye
# (based on the standard 468-point face mesh)
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

def extract_eye_landmarks_from_mediapipe(face_landmarks, image_width, image_height, eye='left'):
    """
    Extract the 6 key points of one eye from MediaPipe face landmarks.

    Parameters
    ----------
    face_landmarks : mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList
        The face landmarks from MediaPipe.
    image_width, image_height : int
        Dimensions of the image.
    eye : str, {'left', 'right'}
        Which eye to extract.

    Returns
    -------
    np.ndarray, shape (6, 2)
        Pixel coordinates of the eye landmarks.
    """
    if eye == 'left':
        indices = LEFT_EYE_INDICES
    else:
        indices = RIGHT_EYE_INDICES

    points = []
    for idx in indices:
        lm = face_landmarks.landmark[idx]
        x = int(lm.x * image_width)
        y = int(lm.y * image_height)
        points.append([x, y])
    return np.array(points, dtype=np.float64)


def extract_both_eyes(face_landmarks, image_width, image_height):
    """
    Extract landmarks for both eyes.

    Returns
    -------
    left_eye, right_eye : np.ndarray, shape (6, 2)
    """
    left = extract_eye_landmarks_from_mediapipe(face_landmarks, image_width, image_height, 'left')
    right = extract_eye_landmarks_from_mediapipe(face_landmarks, image_width, image_height, 'right')
    return left, right