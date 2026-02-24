"""
Pupillary Light Reflex (PLR) analysis.

This module provides functions to analyze the pupil's response to a light stimulus,
including latency, constriction amplitude, maximum speeds, and recovery time.
"""

import numpy as np
from scipy import signal
from typing import Tuple, Optional

from .dynamics import first_derivative


def detect_constriction_onset(
    diameters: np.ndarray,
    times: np.ndarray,
    stimulus_time: float,
    baseline_duration: float = 1.0,
    response_window: float = 2.0,
    speed_threshold: float = -1.0  # px/s, negative
) -> Tuple[Optional[int], Optional[float]]:
    """
    Detect the onset of constriction after a stimulus.

    Parameters
    ----------
    diameters : np.ndarray
        Pupil diameter time series.
    times : np.ndarray
        Corresponding timestamps.
    stimulus_time : float
        Time when the stimulus was applied.
    baseline_duration : float
        Duration before stimulus used to compute baseline (seconds).
    response_window : float
        Time window after stimulus to search for constriction (seconds).
    speed_threshold : float
        Velocity threshold (negative) to consider as constriction start.

    Returns
    -------
    onset_index : int or None
        Index of the first point where constriction starts.
    latency : float or None
        Latency in seconds (stimulus_time to onset).
    """
    # Find indices in the response window
    start_idx = np.searchsorted(times, stimulus_time)
    end_idx = np.searchsorted(times, stimulus_time + response_window)
    if end_idx >= len(times):
        end_idx = len(times) - 1

    if end_idx - start_idx < 3:
        return None, None  # Not enough data

    # Compute velocity in the response window
    vel = first_derivative(diameters, times, smooth=True)
    window_vel = vel[start_idx:end_idx]

    # Find first point where velocity drops below threshold (constriction)
    below_thresh = np.where(window_vel < speed_threshold)[0]
    if len(below_thresh) == 0:
        return None, None

    onset_rel = below_thresh[0]  # first index within window
    onset_abs = start_idx + onset_rel
    latency = times[onset_abs] - stimulus_time

    return onset_abs, latency


def plr_analysis(
    diameters: np.ndarray,
    times: np.ndarray,
    stimulus_time: float,
    baseline_duration: float = 2.0,
    response_window: float = 3.0,
    recovery_level: float = 0.75
) -> dict:
    """
    Perform full pupillary light reflex analysis.

    Parameters
    ----------
    diameters : np.ndarray
        Pupil diameter time series.
    times : np.ndarray
        Corresponding timestamps.
    stimulus_time : float
        Time when the stimulus was applied.
    baseline_duration : float
        Duration before stimulus to compute baseline (seconds).
    response_window : float
        Maximum time after stimulus to search for constriction (seconds).
    recovery_level : float
        Fraction of amplitude for recovery time (e.g., 0.75 for 75% recovery).

    Returns
    -------
    dict
        Dictionary containing:
        - 'baseline_mean': mean diameter before stimulus
        - 'baseline_std': standard deviation during baseline
        - 'min_diameter': minimum diameter after stimulus
        - 'min_index': index of minimum
        - 'amplitude': baseline_mean - min_diameter
        - 'amplitude_percent': (amplitude / baseline_mean) * 100
        - 'latency': time from stimulus to constriction onset (seconds)
        - 'max_constriction_speed': maximum negative velocity during constriction
        - 'max_dilation_speed': maximum positive velocity after minimum
        - 'recovery_time_75': time to reach 75% recovery (from minimum)
        - 'recovery_index_75': index of recovery point
    """
    # Baseline
    baseline_mask = times < stimulus_time
    if np.sum(baseline_mask) < 2:
        raise ValueError("Not enough baseline data")
    baseline_diam = diameters[baseline_mask]
    baseline_mean = np.mean(baseline_diam)
    baseline_std = np.std(baseline_diam, ddof=1)

    # Detect constriction onset
    onset_idx, latency = detect_constriction_onset(
        diameters, times, stimulus_time,
        baseline_duration=baseline_duration,
        response_window=response_window,
        speed_threshold=-1.0  # adjustable
    )

    if onset_idx is None:
        # Fallback: use the global minimum after stimulus as constriction
        post_mask = times > stimulus_time
        if np.sum(post_mask) == 0:
            raise ValueError("No data after stimulus")
        post_idx = np.where(post_mask)[0]
        min_idx_in_post = np.argmin(diameters[post_mask])
        min_idx = post_idx[min_idx_in_post]
    else:
        # Search for minimum after onset
        search_start = onset_idx
        search_end = np.searchsorted(times, stimulus_time + response_window * 2)
        if search_end >= len(times):
            search_end = len(times) - 1
        min_idx = np.argmin(diameters[search_start:search_end]) + search_start

    min_diameter = diameters[min_idx]
    amplitude = baseline_mean - min_diameter
    amplitude_percent = (amplitude / baseline_mean) * 100 if baseline_mean != 0 else 0

    # Velocities
    vel = first_derivative(diameters, times, smooth=True)

    # Maximum constriction speed (most negative) between onset and min
    if onset_idx is not None and min_idx > onset_idx:
        constr_vel = vel[onset_idx:min_idx+1]
        max_constriction_speed = np.min(constr_vel)  # negative
    else:
        max_constriction_speed = 0.0

    # Maximum dilation speed after min (search up to recovery window)
    post_min_start = min_idx + 1
    post_min_end = np.searchsorted(times, times[min_idx] + response_window * 2)
    if post_min_end >= len(times):
        post_min_end = len(times) - 1
    if post_min_start < post_min_end:
        dil_vel = vel[post_min_start:post_min_end+1]
        if len(dil_vel) > 0:
            max_dilation_speed = np.max(dil_vel)
        else:
            max_dilation_speed = 0.0
    else:
        max_dilation_speed = 0.0

    # Recovery time to 75% of amplitude
    target = baseline_mean - recovery_level * amplitude
    # Find first point after min where diameter >= target
    recovery_idx = None
    for i in range(min_idx + 1, len(diameters)):
        if diameters[i] >= target:
            recovery_idx = i
            break
    if recovery_idx is not None:
        recovery_time = times[recovery_idx] - times[min_idx]
    else:
        recovery_time = np.nan

    return {
        'baseline_mean': baseline_mean,
        'baseline_std': baseline_std,
        'min_diameter': min_diameter,
        'min_index': min_idx,
        'amplitude': amplitude,
        'amplitude_percent': amplitude_percent,
        'latency': latency if onset_idx is not None else np.nan,
        'max_constriction_speed': max_constriction_speed,
        'max_dilation_speed': max_dilation_speed,
        'recovery_time_75': recovery_time,
        'recovery_index_75': recovery_idx
    }