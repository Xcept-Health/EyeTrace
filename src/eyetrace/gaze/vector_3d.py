"""
3D gaze vector estimation from eye landmarks.
"""

import numpy as np

def gaze_vector_3d(eye_landmarks: np.ndarray, face_pose=None) -> np.ndarray:
    """
    Estimate 3D gaze direction from eye landmarks.
    
    This is a placeholder; actual implementation requires a geometric eye model.
    Here we return a dummy vector (gaze straight ahead).
    
    Parameters
    ----------
    eye_landmarks : np.ndarray, shape (6, 2)
        2D landmarks of the eye.
    face_pose : optional
        Head pose information (pitch, roll, yaw) to correct gaze.
    
    Returns
    -------
    np.ndarray, shape (3,)
        Gaze direction vector (normalized).
    """
    # Dummy implementation: return vector pointing forward
    return np.array([0, 0, 1])