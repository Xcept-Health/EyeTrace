"""
Neck flexion angle (simplified, requires shoulder detection).
"""

import numpy as np

def neck_flexion_angle(face_landmarks, shoulder_landmarks=None, camera_params=None):
    """
    Estimate neck flexion angle.

    This is a placeholder; actual implementation would require body keypoints
    (e.g., from MediaPipe Pose) to compute the angle between head orientation
    and vertical spine.

    Parameters
    ----------
    face_landmarks : ...
    shoulder_landmarks : optional
    camera_params : optional

    Returns
    -------
    float
        Angle in radians, or np.nan if not available.
    """
    # For now, return a dummy value or raise NotImplementedError
    # You can implement using head pose angles and assuming neutral spine.
    # For simplicity, we'll return 0.0.
    return 0.0