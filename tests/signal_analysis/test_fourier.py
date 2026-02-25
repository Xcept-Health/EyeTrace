"""
Tests for signal_analysis.fourier module.
"""

import pytest
import numpy as np
from eyetrace.signal_analysis.fourier import power_ratio, psd

def test_power_ratio():
    """Test LF/HF power ratio."""
    fs = 100
    t = np.arange(0, 10, 1/fs)
    # Signal avec composantes LF (0.1 Hz) et HF (0.3 Hz)
    x = np.sin(2 * np.pi * 0.1 * t) + 0.5 * np.sin(2 * np.pi * 0.3 * t)
    ratio = power_ratio(x, fs, low_freq=(0.04, 0.15), high_freq=(0.15, 0.4))
    # Le ratio devrait être > 1 car LF domine
    assert ratio > 1.0

def test_psd():
    """Test power spectral density."""
    fs = 100
    t = np.arange(0, 10, 1/fs)
    x = np.sin(2 * np.pi * 5 * t)
    freq, pxx = psd(x, fs)
    # Le pic doit être proche de 5 Hz
    peak_idx = np.argmax(pxx)
    assert np.isclose(freq[peak_idx], 5.0, rtol=0.1)