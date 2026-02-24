"""
Signal-to-noise ratio estimation.
"""

import numpy as np

def signal_to_noise_ratio(signal, method='standard'):
    """
    Estimate SNR of a signal.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    method : {'standard', 'smooth'}
        'standard': SNR = mean / std.
        'smooth': assumes noise is high-frequency, uses smoothed signal as true.

    Returns
    -------
    float
        SNR in linear scale (not dB).
    """
    signal = np.asarray(signal)
    if method == 'standard':
        if np.std(signal) == 0:
            return np.inf
        return np.mean(signal) / np.std(signal)
    elif method == 'smooth':
        from scipy.signal import savgol_filter
        if len(signal) < 10:
            return np.nan
        smooth = savgol_filter(signal, window_length=min(11, len(signal)), polyorder=2)
        noise = signal - smooth
        if np.std(noise) == 0:
            return np.inf
        return np.std(smooth) / np.std(noise)
    else:
        raise ValueError(f"Unknown method: {method}")