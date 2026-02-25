"""
Tests for signal_analysis.fourier module.
"""

import pytest
import numpy as np
from eyetrace.signal_analysis.fourier import lf_hf_ratio, power_spectrum

def test_lf_hf_ratio():
    """Test LF/HF power ratio."""
    fs = 100
    t = np.arange(0, 10, 1/fs)
    # Signal with LF (0.1 Hz) and HF (0.3 Hz) components
    x = np.sin(2 * np.pi * 0.1 * t) + 0.5 * np.sin(2 * np.pi * 0.3 * t)
    ratio = lf_hf_ratio(x, fs, lf_band=(0.04, 0.15), hf_band=(0.15, 0.4))
    # Ratio should be > 1.0 because LF component dominates
    assert ratio > 1.0

def test_power_spectrum():
    """Test power spectral density."""
    fs = 100
    t = np.arange(0, 10, 1/fs)
    x = np.sin(2 * np.pi * 5 * t)
    freq, pxx = power_spectrum(x, fs, method='fft')
    # Peak should be close to 5 Hz
    peak_idx = np.argmax(pxx)
    assert np.isclose(freq[peak_idx], 5.0, rtol=0.1)

    # Test Welch method
    freq2, pxx2 = power_spectrum(x, fs, method='welch')
    assert len(freq2) > 0
    assert len(pxx2) > 0
