"""
Geometric calculations.
"""

import numpy as np


def distance(p1, p2):
    """
    Euclidean distance between two points.

    Parameters
    ----------
    p1, p2 : array-like, shape (2,) or (3,)

    Returns
    -------
    float
        Distance.
    """
    return np.linalg.norm(np.array(p1) - np.array(p2))


def normalize_vector(v):
    """
    Normalize a vector to unit length.

    Parameters
    ----------
    v : array-like

    Returns
    -------
    np.ndarray
        Unit vector (or zero vector if norm is zero).
    """
    norm = np.linalg.norm(v)
    if norm == 0:
        return np.zeros_like(v)
    return np.array(v) / norm


def angle_between_vectors(v1, v2, in_degrees=True):
    """
    Compute the angle between two vectors.

    Parameters
    ----------
    v1, v2 : array-like
    in_degrees : bool, default True
        If True, return angle in degrees; else radians.

    Returns
    -------
    float
        Angle.
    """
    v1_n = normalize_vector(v1)
    v2_n = normalize_vector(v2)
    dot = np.clip(np.dot(v1_n, v2_n), -1.0, 1.0)
    angle_rad = np.arccos(dot)
    if in_degrees:
        return np.degrees(angle_rad)
    return angle_rad


def project_point_to_line(point, line_start, line_end):
    """
    Project a point onto a line defined by two points.

    Parameters
    ----------
    point : array-like
    line_start, line_end : array-like

    Returns
    -------
    np.ndarray
        Projected point.
    """
    p = np.array(point)
    a = np.array(line_start)
    b = np.array(line_end)
    ab = b - a
    ap = p - a
    t = np.dot(ap, ab) / np.dot(ab, ab)
    return a + t * ab