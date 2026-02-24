"""
Pupil eccentricity: angle between optical axis and line of sight.
"""

import numpy as np

def pupil_eccentricity(pupil_center: np.ndarray, iris_center: np.ndarray,
                       gaze_vector: np.ndarray) -> float:
    """
    Estimate pupil eccentricity (simplified).
    
    This is a placeholder; actual computation requires a model of the eye.
    Here we return the angular offset between pupil center and iris center
    as a rough measure.
    
    Parameters
    ----------
    pupil_center : array-like, shape (2,) or (3,)
        Coordinates of pupil center in image or 3D.
    iris_center : array-like, same shape
        Coordinates of iris center.
    gaze_vector : array-like, shape (3,)
        3D gaze direction (optional, not used here).
    
    Returns
    -------
    float
        Eccentricity (radians) – distance between centers normalized by iris radius.
    """
    p = np.asarray(pupil_center)
    i = np.asarray(iris_center)
    if p.ndim == 1 and len(p) == 2:
        # 2D: compute Euclidean distance
        dist = np.linalg.norm(p - i)
        # We need iris radius; assume it's known or pass as parameter
        # For now, return raw distance
        return dist
    else:
        # 3D: angular difference
        p_norm = p / np.linalg.norm(p)
        i_norm = i / np.linalg.norm(i)
        dot = np.clip(np.dot(p_norm, i_norm), -1.0, 1.0)
        return np.arccos(dot)