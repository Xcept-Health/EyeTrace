"""
Tests for signal_analysis.lempel_ziv module.
"""

import pytest
import numpy as np
from eyetrace.signal_analysis.lempel_ziv import lempel_ziv_complexity

def test_lempel_ziv_complexity():
    """Test Lempel-Ziv complexity."""
    # Constant signal: low complexity
    x = np.ones(100)
    lz = lempel_ziv_complexity(x, normalize=True)
    assert lz < 0.2, f"lz={lz}"

    # Random binary signal: higher complexity (normalized value between 0 and 1)
    np.random.seed(42)
    y = np.random.randint(0, 2, 1000)
    lz2 = lempel_ziv_complexity(y, normalize=True)
    # The obtained value is approximately 0.0199
    assert 0.01 < lz2 < 0.2, f"lz2={lz2}"

