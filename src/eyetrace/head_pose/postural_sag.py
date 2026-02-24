"""
Postural sag: change in eye height over time.
"""

import numpy as np

def postural_sag(eye_y_positions, times, baseline_seconds=10):
    """
    Compute postural sag as the linear trend of eye y-coordinate.

    Parameters
    ----------
    eye_y_positions : np.ndarray
        y-coordinate of eye (mean of both eyes) over time.
    times : np.ndarray
        Corresponding timestamps.
    baseline_seconds : float
        Period at start used as baseline.

    Returns
    -------
    slope : float
        Trend slope (pixels per second). Positive = downward sag.
    """
    if len(eye_y_positions) < 2:
        return np.nan
    # Linear regression
    t = times - times[0]
    y = eye_y_positions
    slope = np.polyfit(t, y, 1)[0]
    return slope