"""
Tests for signal_analysis.higuchi module.
"""

import pytest
import numpy as np
from eyetrace.signal_analysis.higuchi import higuchi_fd

def test_higuchi_fd():
    """Test Higuchi fractal dimension."""
    # Signal linéaire : dimension fractale proche de 1
    x = np.linspace(0, 10, 1000)
    fd = higuchi_fd(x, kmax=10)
    assert np.isclose(fd, 1.0, rtol=0.1)

    # Signal bruit blanc : dimension fractale proche de 2
    np.random.seed(42)
    y = np.random.randn(1000)
    fd2 = higuchi_fd(y, kmax=10)
    assert fd2 > 1.5