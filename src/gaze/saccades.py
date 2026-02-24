"""
Saccade detection and analysis.
"""

import numpy as np
from scipy.ndimage import label
from typing import List, Tuple, Optional
from .utils import angular_velocity

def saccade_velocity(gaze_positions: np.ndarray, times: np.ndarray,
                     smooth: bool = True) -> np.ndarray:
    """
    Compute instantaneous angular velocity of gaze.
    
    Parameters
    ----------
    gaze_positions : np.ndarray, shape (n, 3)
        3D gaze vectors (can be unnormalized).
    times : np.ndarray, shape (n,)
        Timestamps.
    smooth : bool
        Whether to apply Savitzky-Golay smoothing to positions.
    
    Returns
    -------
    np.ndarray, shape (n,)
        Angular velocity in radians per second.
    """
    if len(gaze_positions) < 2:
        return np.array([])
    
    if smooth and len(gaze_positions) >= 5:
        from scipy.signal import savgol_filter
        # Smooth each coordinate separately
        smoothed = np.zeros_like(gaze_positions)
        for i in range(3):
            smoothed[:, i] = savgol_filter(gaze_positions[:, i],
                                           window_length=min(5, len(gaze_positions)),
                                           polyorder=2)
        gaze_positions = smoothed
    
    velocities = []
    for i in range(1, len(gaze_positions)):
        dt = times[i] - times[i-1]
        vel = angular_velocity(gaze_positions[i-1], gaze_positions[i], dt)
        velocities.append(vel)
    # Duplicate last value to keep same length
    velocities.append(velocities[-1] if velocities else 0)
    return np.array(velocities)


def saccade_acceleration(velocities: np.ndarray, times: np.ndarray) -> np.ndarray:
    """
    Compute angular acceleration from velocity time series.
    """
    if len(velocities) < 2:
        return np.array([])
    acc = np.gradient(velocities, times)
    return acc


def detect_saccades(gaze_positions: np.ndarray, times: np.ndarray,
                    velocity_threshold: float = 30.0,  # deg/s, à ajuster
                    min_duration: float = 0.01,        # secondes
                    merge_interval: float = 0.05) -> List[Tuple[int, int]]:
    """
    Detect saccades based on velocity threshold.
    
    Parameters
    ----------
    gaze_positions : np.ndarray, shape (n, 3)
        3D gaze vectors.
    times : np.ndarray, shape (n,)
        Timestamps.
    velocity_threshold : float
        Angular velocity threshold (in deg/s or rad/s – attention unit).
    min_duration : float
        Minimum saccade duration (s).
    merge_interval : float
        If two saccades are closer than this, merge them.
    
    Returns
    -------
    List[Tuple[int, int]]
        List of (start_idx, end_idx) for each saccade.
    """
    # Convert threshold to rad/s if needed (assuming input in rad/s)
    # On suppose que velocity_threshold est en rad/s (cohérent avec angular_velocity)
    vel = saccade_velocity(gaze_positions, times, smooth=True)
    is_saccade = vel > velocity_threshold
    
    labeled, n_features = label(is_saccade)
    saccades = []
    for region_id in range(1, n_features + 1):
        region = np.where(labeled == region_id)[0]
        duration = times[region[-1]] - times[region[0]]
        if duration >= min_duration:
            saccades.append((region[0], region[-1]))
    
    # Merge close saccades
    if not saccades:
        return []
    merged = [saccades[0]]
    for s in saccades[1:]:
        prev_end = merged[-1][1]
        if times[s[0]] - times[prev_end] <= merge_interval:
            merged[-1] = (merged[-1][0], s[1])
        else:
            merged.append(s)
    return merged


def saccade_fixation_ratio(saccades: List[Tuple[int, int]], total_frames: int) -> float:
    """
    Compute ratio of time spent in saccades vs fixations.
    
    Parameters
    ----------
    saccades : List[Tuple[int, int]]
        Detected saccade intervals.
    total_frames : int
        Total number of frames.
    
    Returns
    -------
    float
        Ratio saccade_time / fixation_time (or NaN if no fixation).
    """
    saccade_frames = 0
    for start, end in saccades:
        saccade_frames += (end - start + 1)
    fixation_frames = total_frames - saccade_frames
    if fixation_frames == 0:
        return np.nan
    return saccade_frames / fixation_frames