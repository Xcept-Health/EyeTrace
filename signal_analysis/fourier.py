"""
Frequency domain analysis using FFT.
"""

import numpy as np
from scipy.signal import welch

def power_spectrum(signal, fs, method='fft'):
    """
    Compute the power spectrum of a signal.

    Parameters
    ----------
    signal : np.ndarray
        Input signal (1D).
    fs : float
        Sampling frequency in Hz.
    method : {'fft', 'welch'}
        'fft': simple FFT (magnitude squared).
        'welch': Welch's averaged periodogram (less noisy).

    Returns
    -------
    freqs : np.ndarray
        Frequency array.
    power : np.ndarray
        Power spectral density.
    """
    signal = np.asarray(signal)
    if method == 'fft':
        n = len(signal)
        fft_vals = np.fft.rfft(signal)
        power = np.abs(fft_vals) ** 2 / n
        freqs = np.fft.rfftfreq(n, d=1/fs)
        return freqs, power
    elif method == 'welch':
        freqs, psd = welch(signal, fs, nperseg=min(256, len(signal)))
        return freqs, psd
    else:
        raise ValueError(f"Unknown method: {method}")


def band_power(freqs, power, band):
    """
    Integrate power over a frequency band.

    Parameters
    ----------
    freqs : np.ndarray
        Frequency array.
    power : np.ndarray
        Power (or PSD) array.
    band : tuple (low, high)
        Frequency band of interest.

    Returns
    -------
    float
        Total power in the band.
    """
    low, high = band
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return 0.0
    # If power is PSD, multiply by frequency resolution
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    return np.trapz(power[mask], dx=df)


def lf_hf_ratio(signal, fs, lf_band=(0.04, 0.15), hf_band=(0.15, 0.4)):
    """
    Compute the LF/HF ratio, often used in HRV but applicable to pupil signals.

    Parameters
    ----------
    signal : np.ndarray
        Input signal (e.g., pupil diameter).
    fs : float
        Sampling frequency.
    lf_band, hf_band : tuple
        Frequency bands for low and high frequencies.

    Returns
    -------
    float
        Ratio LF/HF.
    """
    freqs, psd = power_spectrum(signal, fs, method='welch')
    lf = band_power(freqs, psd, lf_band)
    hf = band_power(freqs, psd, hf_band)
    if hf == 0:
        return np.inf
    return lf / hf