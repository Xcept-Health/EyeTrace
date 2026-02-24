"""
PERCLOS (Percentage of Eyelid Closure) calculation.
"""

import numpy as np

def perclos(ear_sequence: np.ndarray, threshold: float = 0.2,
            window_seconds: float = 60.0, frame_rate: float = 30.0) -> float:
    """
    Compute PERCLOS over a sliding window (or the whole sequence).

    PERCLOS is the percentage of time the eyes are closed (EAR < threshold)
    over a given window. Typically, a 60-second window is used.

    Parameters
    ----------
    ear_sequence : np.ndarray
        Array of EAR values.
    threshold : float
        EAR threshold for closed eye.
    window_seconds : float
        Duration of the window in seconds. If the sequence is shorter,
        the whole sequence is used.
    frame_rate : float
        Frame rate (fps) to convert window_seconds to frames.

    Returns
    -------
    float
        PERCLOS value (percentage, 0-100).
    """
    window_frames = int(window_seconds * frame_rate)
    if len(ear_sequence) <= window_frames:
        closed = ear_sequence < threshold
        return np.mean(closed) * 100.0
    else:
        # Use last `window_frames` frames
        closed = ear_sequence[-window_frames:] < threshold
        return np.mean(closed) * 100.0