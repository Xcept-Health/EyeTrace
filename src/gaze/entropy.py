"""
Gaze entropy (Shannon) based on spatial discretization.
"""

import numpy as np

def gaze_entropy(gaze_positions: np.ndarray, bins: int = 10,
                 range_x: tuple = None, range_y: tuple = None) -> float:
    """
    Compute Shannon entropy of gaze positions.
    
    The space is discretized into a 2D histogram; entropy is computed from
    the probability distribution.
    
    Parameters
    ----------
    gaze_positions : np.ndarray, shape (n, 2)
        2D gaze positions (e.g., screen coordinates).
    bins : int or tuple
        Number of bins per dimension (if int, same for x and y).
    range_x, range_y : tuple, optional
        (min, max) for each axis. If None, use data range.
    
    Returns
    -------
    float
        Entropy in bits (using log2).
    """
    if gaze_positions.shape[1] != 2:
        raise ValueError("gaze_entropy requires 2D positions")
    
    if range_x is None:
        range_x = (np.min(gaze_positions[:, 0]), np.max(gaze_positions[:, 0]))
    if range_y is None:
        range_y = (np.min(gaze_positions[:, 1]), np.max(gaze_positions[:, 1]))
    
    hist, xedges, yedges = np.histogram2d(
        gaze_positions[:, 0], gaze_positions[:, 1],
        bins=bins, range=[range_x, range_y]
    )
    prob = hist / np.sum(hist)
    prob = prob[prob > 0]  # remove zeros
    entropy = -np.sum(prob * np.log2(prob))
    return entropy