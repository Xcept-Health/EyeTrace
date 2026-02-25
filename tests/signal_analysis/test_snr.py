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
    
    # Use the 'smooth' method which estimates SNR from the noisy signal
    snr = signal_to_noise_ratio(noisy, method='smooth')
    # Expected linear SNR is around 10 (signal amplitude ~1, noise ~0.1)
    assert 5 < snr < 20, f"snr={snr}"

    # Test the 'standard' method on a constant signal
    const = np.ones(100) * 5
    snr2 = signal_to_noise_ratio(const, method='standard')
    # Check for infinity or high value depending on zero-division handling
    assert snr2 == np.inf or snr2 > 100
