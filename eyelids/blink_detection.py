"""
Blink detection and related metrics.
"""

import numpy as np
from scipy.ndimage import label
from typing import List, Tuple, Optional

def detect_blinks(ear_sequence: np.ndarray, threshold: float = 0.2,
                  min_interval_frames: int = 10) -> List[Tuple[int, int]]:
    """
    Detect blinks in an EAR time series.

    A blink is defined as a period where EAR drops below `threshold`
    and then rises above it. The function returns a list of (start, end)
    indices for each blink.

    Parameters
    ----------
    ear_sequence : np.ndarray
        1D array of EAR values.
    threshold : float
        EAR threshold below which the eye is considered closed.
    min_interval_frames : int
        Minimum number of frames between two blinks to merge them.

    Returns
    -------
    List[Tuple[int, int]]
        List of (start, end) indices for each blink.
    """
    closed = ear_sequence < threshold
    labeled, n_features = label(closed)

    blinks = []
    for region_id in range(1, n_features + 1):
        region = np.where(labeled == region_id)[0]
        if len(region) > 1:  # at least 2 frames to be a blink
            blinks.append((region[0], region[-1]))

    # Merge if too close (optional)
    merged = []
    for blink in blinks:
        if not merged or blink[0] - merged[-1][1] > min_interval_frames:
            merged.append(blink)
        else:
            # Merge with previous
            merged[-1] = (merged[-1][0], blink[1])
    return merged


def blink_frequency(blinks: List[Tuple[int, int]], duration_seconds: float) -> float:
    """
    Compute blink frequency in blinks per minute.

    Parameters
    ----------
    blinks : List[Tuple[int, int]]
        List of blink intervals (from detect_blinks).
    duration_seconds : float
        Total duration of the recording in seconds.

    Returns
    -------
    float
        Blink frequency (blinks per minute).
    """
    if duration_seconds <= 0:
        return 0.0
    return len(blinks) * 60.0 / duration_seconds


def mean_closure_duration(blinks: List[Tuple[int, int]], frame_rate: float) -> float:
    """
    Compute mean closure duration (MCD) in seconds.

    Parameters
    ----------
    blinks : List[Tuple[int, int]]
        List of blink intervals.
    frame_rate : float
        Frame rate of the recording (fps).

    Returns
    -------
    float
        Mean closure duration in seconds.
    """
    if not blinks:
        return 0.0
    durations = [(end - start + 1) / frame_rate for start, end in blinks]
    return np.mean(durations)


def long_blink_ratio(blinks: List[Tuple[int, int]], frame_rate: float,
                     threshold_seconds: float = 0.5) -> float:
    """
    Ratio of blinks longer than a threshold to total blinks.

    Parameters
    ----------
    blinks : List[Tuple[int, int]]
        List of blink intervals.
    frame_rate : float
        Frame rate.
    threshold_seconds : float
        Duration threshold for long blinks (e.g., 0.5 s).

    Returns
    -------
    float
        Ratio (0 to 1).
    """
    if not blinks:
        return 0.0
    durations = [(end - start + 1) / frame_rate for start, end in blinks]
    long_count = sum(1 for d in durations if d > threshold_seconds)
    return long_count / len(blinks)