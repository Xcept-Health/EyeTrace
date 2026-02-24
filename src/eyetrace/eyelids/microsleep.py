"""
Microsleep indicator: periods of prolonged eye closure.
"""

import numpy as np
from .blink_detection import detect_blinks

def microsleep_indicator(ear_sequence: np.ndarray, frame_rate: float,
                         ear_threshold: float = 0.2,
                         duration_threshold: float = 2.0) -> np.ndarray:
    """
    Detect microsleep events (prolonged eye closure).

    A microsleep is defined as a period where EAR is below threshold
    for at least `duration_threshold` seconds.

    Parameters
    ----------
    ear_sequence : np.ndarray
        EAR time series.
    frame_rate : float
        Frame rate (fps).
    ear_threshold : float
        EAR threshold for closed eye.
    duration_threshold : float
        Minimum duration (seconds) to qualify as microsleep.

    Returns
    -------
    np.ndarray
        Boolean array of same length, True during microsleep periods.
    """
    closed = ear_sequence < ear_threshold
    # Label contiguous regions
    from scipy.ndimage import label
    labeled, n_features = label(closed)

    microsleep = np.zeros_like(closed, dtype=bool)
    min_frames = int(duration_threshold * frame_rate)
    for region_id in range(1, n_features + 1):
        region = np.where(labeled == region_id)[0]
        if len(region) >= min_frames:
            microsleep[region] = True
    return microsleep