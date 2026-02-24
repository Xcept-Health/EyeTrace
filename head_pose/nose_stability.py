"""
Nose stability (variance of nose position) as an indicator of head movement.
"""

import numpy as np

def nose_stability(face_landmarks_sequence, image_width, image_height):
    """
    Compute variance of nose tip position over time.

    Parameters
    ----------
    face_landmarks_sequence : list of mediapipe landmarks
        List of face landmarks per frame.
    image_width, image_height : int

    Returns
    -------
    float
        Variance (in pixels^2) of nose tip position.
    """
    nose_positions = []
    for landmarks in face_landmarks_sequence:
        nose = landmarks.landmark[1]  # index 1 is nose tip
        x = nose.x * image_width
        y = nose.y * image_height
        nose_positions.append([x, y])
    nose_positions = np.array(nose_positions)
    if len(nose_positions) < 2:
        return np.nan
    return np.var(nose_positions, axis=0).mean()  # mean variance across x and y