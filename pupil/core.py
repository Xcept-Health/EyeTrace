"""
Core functions for extracting pupil and iris measurements from MediaPipe landmarks.
"""

import numpy as np
from typing import Tuple, Optional

def extract_pupil_diameter(
    iris_landmarks: np.ndarray,
    image_width: int,
    image_height: int,
    px_to_mm: Optional[float] = None
) -> float:
    """
    Estimate pupil diameter from iris landmarks.

    MediaPipe provides 5 landmarks per iris (indices 468-472 for left, 473-477 for right).
    We approximate the pupil by the iris center and assume a fixed ratio (pupil/iris)
    or use a more sophisticated method. For simplicity, we compute the diameter as
    the mean distance from the iris center to the landmarks, then multiply by 2.

    Parameters
    ----------
    iris_landmarks : np.ndarray, shape (5, 2)
        (x, y) coordinates of the 5 iris landmarks.
    image_width : int
        Width of the original image (for scaling if needed).
    image_height : int
        Height of the original image.
    px_to_mm : float, optional
        Conversion factor from pixels to millimeters. If None, returns diameter in pixels.

    Returns
    -------
    float
        Estimated pupil diameter (in pixels or mm).
    """
    if iris_landmarks.shape != (5, 2):
        raise ValueError(f"Expected (5,2) landmarks, got {iris_landmarks.shape}")

    # Compute iris center as mean of landmarks
    center = np.mean(iris_landmarks, axis=0)

    # Compute mean distance from center to landmarks (approximate radius)
    distances = np.linalg.norm(iris_landmarks - center, axis=1)
    radius = np.mean(distances)

    # Diameter = 2 * radius
    diameter_px = 2 * radius

    if px_to_mm is not None:
        return diameter_px * px_to_mm
    return diameter_px


def extract_iris_radius(
    iris_landmarks: np.ndarray,
    image_width: int,
    image_height: int,
    px_to_mm: Optional[float] = None
) -> float:
    """
    Estimate iris radius from iris landmarks.

    Parameters
    ----------
    iris_landmarks : np.ndarray, shape (5, 2)
        (x, y) coordinates of the 5 iris landmarks.
    image_width, image_height : int
        Image dimensions (unused here but kept for API consistency).
    px_to_mm : float, optional
        Conversion factor.

    Returns
    -------
    float
        Iris radius (pixels or mm).
    """
    if iris_landmarks.shape != (5, 2):
        raise ValueError(f"Expected (5,2) landmarks, got {iris_landmarks.shape}")

    center = np.mean(iris_landmarks, axis=0)
    distances = np.linalg.norm(iris_landmarks - center, axis=1)
    radius = np.mean(distances)

    if px_to_mm is not None:
        return radius * px_to_mm
    return radius