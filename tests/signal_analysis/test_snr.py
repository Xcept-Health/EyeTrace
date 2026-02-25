"""
Tests for signal_analysis.snr module.
"""

import pytest
import numpy as np
from eyetrace.signal_analysis.snr import signal_to_noise_ratio

def test_signal_to_noise_ratio():
    """Test SNR calculation."""
    t = np.linspace(0, 1, 1000)
    clean = np.sin(2 * np.pi * 10 * t)
    noisy = clean + 0.1 * np.random.randn(1000)
    snr = signal_to_noise_ratio(clean, noisy)
    # SNR environ 20 dB (car amplitude bruit 0.1, signal 1)
    assert 10 < snr < 30