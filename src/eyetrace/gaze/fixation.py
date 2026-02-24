"""
Fixation analysis: duration, dispersion, centroid.
"""

import numpy as np
from scipy.ndimage import label
from typing import List, Tuple

def fixation_duration(fixation_mask: np.ndarray, times: np.ndarray) -> List[float]:
    """
    Compute duration of each fixation from a boolean mask.
    
    Parameters
    ----------
    fixation_mask : np.ndarray, bool
        True for frames classified as fixation.
    times : np.ndarray
        Timestamps.
    
    Returns
    -------
    List[float]
        Durations (seconds) of each fixation.
    """
    labeled, n_features = label(fixation_mask)
    durations = []
    for region_id in range(1, n_features + 1):
        region = np.where(labeled == region_id)[0]
        if len(region) > 0:
            dur = times[region[-1]] - times[region[0]]
            durations.append(dur)
    return durations


def fixation_dispersion(gaze_positions: np.ndarray, fixation_mask: np.ndarray) -> List[float]:
    """
    Compute spatial dispersion (standard deviation) for each fixation.
    
    Parameters
    ----------
    gaze_positions : np.ndarray, shape (n, 2) or (n, 3)
        Gaze positions (2D or 3D).
    fixation_mask : np.ndarray, bool
        True for frames in fixation.
    
    Returns
    -------
    List[float]
        Dispersion (std dev) for each fixation.
    """
    labeled, n_features = label(fixation_mask)
    dispersions = []
    for region_id in range(1, n_features + 1):
        region = np.where(labeled == region_id)[0]
        if len(region) > 1:
            points = gaze_positions[region]
            if points.shape[1] == 2:
                # 2D: mean of std in x and y
                disp = np.mean(np.std(points, axis=0))
            else:
                # 3D: mean of std in x,y,z
                disp = np.mean(np.std(points, axis=0))
            dispersions.append(disp)
        elif len(region) == 1:
            dispersions.append(0.0)
    return dispersions


def gaze_centroid(gaze_positions: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """
    Compute the centroid (mean) of gaze positions.
    
    Parameters
    ----------
    gaze_positions : np.ndarray, shape (n, 2) or (n, 3)
    mask : np.ndarray, bool, optional
        If provided, only use frames where mask is True.
    
    Returns
    -------
    np.ndarray
        Mean position.
    """
    if mask is not None:
        points = gaze_positions[mask]
    else:
        points = gaze_positions
    if len(points) == 0:
        return np.full(gaze_positions.shape[1], np.nan)
    return np.mean(points, axis=0)