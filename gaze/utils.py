"""
Utility functions for gaze module, including coordinate transformations.
"""

import numpy as np

def cartesian_to_spherical(x, y, z):
    """
    Convert 3D Cartesian coordinates to spherical coordinates (theta, phi, r).
    
    Parameters
    ----------
    x, y, z : float or np.ndarray
        Cartesian coordinates.
    
    Returns
    -------
    theta : float or np.ndarray
        Azimuth angle (radians, from -pi to pi).
    phi : float or np.ndarray
        Elevation angle (radians, from -pi/2 to pi/2).
    r : float or np.ndarray
        Radius.
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)
    phi = np.arcsin(z / r) if r != 0 else 0
    return theta, phi, r


def spherical_to_cartesian(theta, phi, r=1.0):
    """
    Convert spherical coordinates to Cartesian.
    """
    x = r * np.cos(phi) * np.cos(theta)
    y = r * np.cos(phi) * np.sin(theta)
    z = r * np.sin(phi)
    return x, y, z


def angular_velocity(pos1, pos2, dt):
    """
    Compute angular velocity between two 3D gaze vectors.
    
    Parameters
    ----------
    pos1, pos2 : array-like, shape (3,)
        Gaze vectors (normalized or not).
    dt : float
        Time difference.
    
    Returns
    -------
    float
        Angular velocity in radians per second.
    """
    v1 = np.asarray(pos1)
    v2 = np.asarray(pos2)
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
    angle = np.arccos(cos_angle)
    return angle / dt if dt > 0 else 0.0