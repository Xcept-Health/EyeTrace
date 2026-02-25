"""
Yawning detection based on MAR.
"""

import numpy as np
from scipy.ndimage import label


def yawn_detection(mar_sequence, threshold=0.6, min_duration=1.0, frame_rate=30):
    """
    Detect yawns from MAR time series.

    Parameters
    ----------
    mar_sequence : np.ndarray
        MAR values.
    threshold : float
        MAR threshold to consider mouth open.
    min_duration : float
        Minimum duration (seconds) to qualify as yawn.
    frame_rate : float
        Frame rate.

    Returns
    -------
    list of (start, end) indices.
    """
    is_open = mar_sequence > threshold
    labeled, n_features = label(is_open)
    yawns = []
    min_frames = int(min_duration * frame_rate)
    for region_id in range(1, n_features+1):
        region = np.where(labeled == region_id)[0]
        if len(region) >= min_frames:
            yawns.append((region[0], region[-1]))
    return yawns


def yawn_frequency(yawns, total_duration):
    """
    Yawns per minute.
    """
    if total_duration <= 0:
        return 0.0
    return len(yawns) * 60.0 / total_duration