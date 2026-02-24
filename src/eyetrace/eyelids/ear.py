"""
Eye Aspect Ratio (EAR) calculation.
"""

import numpy as np

def eye_aspect_ratio(eye_landmarks):
    """
    Calculate the Eye Aspect Ratio (EAR) from eye landmarks.

    Parameters
    ----------
    eye_landmarks : np.ndarray, shape (6, 2)
        Coordinates of the 6 eye landmarks in the following order:
        [p1, p2, p3, p4, p5, p6] where:
            p1: outer corner of the eye
            p2: top outer point
            p3: top inner point
            p4: inner corner of the eye
            p5: bottom inner point
            p6: bottom outer point

    Returns
    -------
    float
        The EAR value.
    """
    if eye_landmarks.shape != (6, 2):
        raise ValueError(f"Expected (6, 2) landmarks, got {eye_landmarks.shape}")

    # Compute vertical distances
    vertical_1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    vertical_2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])

    # Compute horizontal distance
    horizontal = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def both_eyes_ear(left_eye, right_eye):
    """
    Calculate the average EAR for both eyes.

    Parameters
    ----------
    left_eye, right_eye : np.ndarray, shape (6, 2)
        Landmarks for left and right eyes.

    Returns
    -------
    float
        Average EAR.
    """
    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)
    return (left_ear + right_ear) / 2.0