"""
Tests for signal_analysis.entropy module.
"""

import pytest
import numpy as np
from eyetrace.signal_analysis.entropy import sample_entropy

def test_sample_entropy():
    """Test sample entropy calculation."""
    # Signal régulier : faible entropie
    t = np.linspace(0, 10, 1000)
    x = np.sin(2 * np.pi * 1 * t)  # sinusoïde pure
    se = sample_entropy(x, m=2, r=0.2)
    # Une sinusoïde pure a une entropie basse (proche de 0)
    assert se < 0.5

    # Signal aléatoire : entropie plus élevée
    np.random.seed(42)
    y = np.random.randn(1000)
    se_rand = sample_entropy(y, m=2, r=0.2)
    assert se_rand > 0.5

def test_sample_entropy_short():
    """Test with too short data."""
    x = np.array([1, 2])
    se = sample_entropy(x, m=2, r=0.2)
    assert np.isnan(se) or se == 0.0  # selon implémentation