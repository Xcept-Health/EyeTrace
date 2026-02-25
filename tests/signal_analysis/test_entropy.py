"""
Tests for signal_analysis.entropy module.
"""

import pytest
import numpy as np
from eyetrace.signal_analysis.entropy import sample_entropy

def test_sample_entropy():
    """Test sample entropy calculation."""
    # Regular signal: low entropy
    t = np.linspace(0, 10, 1000)
    x = np.sin(2 * np.pi * 1 * t)  # pure sinusoid
    se = sample_entropy(x, m=2, r=0.2)
    # A pure sinusoid has low entropy (close to 0)
    assert se < 0.5

    # Random signal: higher entropy
    np.random.seed(42)
    y = np.random.randn(1000)
    se_rand = sample_entropy(y, m=2, r=0.2)
    assert se_rand > 0.5

def test_sample_entropy_short():
    """Test with data that is too short."""
    x = np.array([1, 2])
    se = sample_entropy(x, m=2, r=0.2)
    # Depending on implementation, should return NaN or 0.0
    assert np.isnan(se) or se == 0.0
