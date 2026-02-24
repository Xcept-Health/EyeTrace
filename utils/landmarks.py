"""
Landmark extraction from MediaPipe face mesh.
"""

import numpy as np

# MediaPipe indices for eyes (6 points each)
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# MediaPipe indices for iris (5 points each)
LEFT_IRIS_INDICES = list(range(468, 473))
RIGHT_IRIS_INDICES = list(range(473, 478))

# Indices for other facial features (can be extended)
NOSE_TIP = 1
CHIN = 152
LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
RIGHT_EYE_OUTER = 362
RIGHT_EYE_INNER = 263
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
MOUTH_TOP = 13
MOUTH_BOTTOM = 14


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


def extract_iris_landmarks_from_mediapipe(face_landmarks, image_width, image_height, eye='left'):
    """
    Extract the 5 iris points from MediaPipe face landmarks.

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
    np.ndarray, shape (5, 2)
        Pixel coordinates of the iris landmarks.
    """
    if eye == 'left':
        indices = LEFT_IRIS_INDICES
    else:
        indices = RIGHT_IRIS_INDICES

    points = []
    for idx in indices:
        lm = face_landmarks.landmark[idx]
        x = int(lm.x * image_width)
        y = int(lm.y * image_height)
        points.append([x, y])
    return np.array(points, dtype=np.float64)


def extract_face_landmarks_array(face_landmarks, image_width, image_height, indices=None):
    """
    Extract a set of face landmarks as a 2D array.

    Parameters
    ----------
    face_landmarks : mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList
        The face landmarks from MediaPipe.
    image_width, image_height : int
        Dimensions of the image.
    indices : list of int, optional
        List of landmark indices to extract. If None, returns all 468 points.

    Returns
    -------
    np.ndarray, shape (n_landmarks, 2)
        Pixel coordinates.
    """
    if indices is None:
        indices = range(len(face_landmarks.landmark))

    points = []
    for idx in indices:
        lm = face_landmarks.landmark[idx]
        x = int(lm.x * image_width)
        y = int(lm.y * image_height)
        points.append([x, y])
    return np.array(points, dtype=np.float64)