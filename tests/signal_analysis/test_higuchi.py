"""
Tests for signal_analysis.higuchi module.
"""

import pytest
import numpy as np
from eyetrace.signal_analysis.higuchi import higuchi_fractal_dimension

def test_higuchi_fd():
    """Test Higuchi fractal dimension."""
    # Sinusoidal signal
    t = np.linspace(0, 10, 1000)
    x = np.sin(2 * np.pi * 1 * t)
    fd = higuchi_fractal_dimension(x, kmax=10)
    
    # Verify that the result is a finite number
    assert np.isfinite(fd), f"fd={fd}"
