"""
Dynamic pupil metrics: constriction/dilation speeds, first derivative, hippus.
"""

import numpy as np
from scipy import signal
from typing import Tuple, Optional


def first_derivative(
    diameters: np.ndarray,
    times: Optional[np.ndarray] = None,
    smooth: bool = True,
    window_length: int = 5,
    polyorder: int = 2
) -> np.ndarray:
    """
    Compute the first derivative (rate of change) of pupil diameter.

    If times are not provided, assumes uniform sampling (dt=1). The derivative
    is computed using central differences, optionally smoothed with Savitzky-Golay.

    Parameters
    ----------
    diameters : np.ndarray
        1D array of pupil diameters.
    times : np.ndarray, optional
        1D array of timestamps (same length). If None, assumes uniform sampling.
    smooth : bool, default True
        Whether to apply Savitzky-Golay smoothing before differentiation.
    window_length : int, default 5
        Window length for Savitzky-Golay (must be odd).
    polyorder : int, default 2
        Polynomial order for Savitzky-Golay.

    Returns
    -------
    np.ndarray
        First derivative (velocity) at each point.
    """
    arr = np.asarray(diameters, dtype=np.float64)
    if smooth:
        from scipy.signal import savgol_filter
        if len(arr) >= window_length:
            arr = savgol_filter(arr, window_length, polyorder)
        # else: not enough points for smoothing

    if times is None:
        dt = 1.0
    else:
        times = np.asarray(times, dtype=np.float64)
        dt = np.mean(np.diff(times))

    deriv = np.gradient(arr, dt)
    return deriv


def constriction_speed(
    diameters: np.ndarray,
    times: np.ndarray,
    threshold: float = 0.5,
    min_duration: float = 0.1
) -> Tuple[float, float]:
    """
    Compute the maximum constriction speed and the average speed during the
    main constriction phase.

    Parameters
    ----------
    diameters : np.ndarray
        Pupil diameters.
    times : np.ndarray
        Corresponding timestamps.
    threshold : float
        Minimum absolute speed to consider as constriction (to ignore noise).
    min_duration : float
        Minimum duration of constriction phase (seconds).

    Returns
    -------
    max_speed : float
        Maximum constriction speed (negative value).
    avg_speed : float
        Average speed during the constriction phase.
    """
    vel = first_derivative(diameters, times, smooth=True)
    is_constricting = vel < -threshold
    if not np.any(is_constricting):
        return 0.0, 0.0

    from scipy.ndimage import label
    labeled, n_features = label(is_constricting)

    best_region = None
    best_min_vel = 0.0
    for region_id in range(1, n_features + 1):
        mask = labeled == region_id
        region_vel = vel[mask]
        if len(region_vel) * (times[1] - times[0]) < min_duration:
            continue
        min_vel = np.min(region_vel)
        if min_vel < best_min_vel:
            best_min_vel = min_vel
            best_region = mask

    if best_region is None:
        return 0.0, 0.0

    max_speed = np.min(vel[best_region])
    avg_speed = np.mean(vel[best_region])
    return max_speed, avg_speed


def dilation_speed(
    diameters: np.ndarray,
    times: np.ndarray,
    threshold: float = 0.5,
    min_duration: float = 0.1
) -> Tuple[float, float]:
    """
    Compute the maximum dilation speed and average speed during dilation.

    Parameters
    ----------
    diameters : np.ndarray
        Pupil diameters.
    times : np.ndarray
        Corresponding timestamps.
    threshold : float
        Minimum speed to consider as dilation.
    min_duration : float
        Minimum duration of dilation phase (seconds).

    Returns
    -------
    max_speed : float
        Maximum dilation speed.
    avg_speed : float
        Average speed during dilation.
    """
    vel = first_derivative(diameters, times, smooth=True)
    is_dilating = vel > threshold
    if not np.any(is_dilating):
        return 0.0, 0.0

    from scipy.ndimage import label
    labeled, n_features = label(is_dilating)

    best_region = None
    best_max_vel = 0.0
    for region_id in range(1, n_features + 1):
        mask = labeled == region_id
        region_vel = vel[mask]
        if len(region_vel) * (times[1] - times[0]) < min_duration:
            continue
        max_vel = np.max(region_vel)
        if max_vel > best_max_vel:
            best_max_vel = max_vel
            best_region = mask

    if best_region is None:
        return 0.0, 0.0

    max_speed = np.max(vel[best_region])
    avg_speed = np.mean(vel[best_region])
    return max_speed, avg_speed


def hippus_amplitude(
    diameters: np.ndarray,
    fs: float,
    lowcut: float = 0.5,
    highcut: float = 4.0,
    method: str = 'rms'
) -> float:
    """
    Compute the amplitude of pupillary hippus (spontaneous oscillations).

    The signal is bandpass filtered in the hippus frequency range (typically 0.5-4 Hz),
    then the amplitude is estimated as either the root mean square (RMS) or
    the mean of the envelope.

    Parameters
    ----------
    diameters : np.ndarray
        Pupil diameters (must be uniformly sampled).
    fs : float
        Sampling frequency in Hz.
    lowcut : float, default 0.5
        Lower cutoff frequency for bandpass filter.
    highcut : float, default 4.0
        Upper cutoff frequency.
    method : {'rms', 'envelope'}, default 'rms'
        Method to estimate amplitude:
        - 'rms': root mean square of filtered signal.
        - 'envelope': mean of the Hilbert envelope.

    Returns
    -------
    float
        Hippus amplitude (in same units as diameters).
    """
    from scipy.signal import butter, filtfilt, hilbert

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if high >= 1.0:
        raise ValueError(f"High cutoff {highcut} Hz must be less than Nyquist {nyq} Hz")
    b, a = butter(4, [low, high], btype='band')

    filtered = filtfilt(b, a, diameters)

    if method == 'rms':
        return np.sqrt(np.mean(filtered**2))
    elif method == 'envelope':
        analytic = hilbert(filtered)
        envelope = np.abs(analytic)
        return np.mean(envelope)
    else:
        raise ValueError(f"Unknown method: {method}")