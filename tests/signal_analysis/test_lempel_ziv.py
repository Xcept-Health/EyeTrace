"""
Tests for signal_analysis.lempel_ziv module.
"""

import pytest
import numpy as np
from eyetrace.signal_analysis.lempel_ziv import lempel_ziv_complexity

def test_lempel_ziv_complexity():
    """Test Lempel-Ziv complexity."""
    # Signal constant : complexité faible
    x = np.ones(100)
    lz = lempel_ziv_complexity(x)
    assert lz < 10

    # Signal aléatoire : complexité élevée
    np.random.seed(42)
    y = np.random.randint(0, 2, 1000)  # binaire
    lz2 = lempel_ziv_complexity(y)
    assert lz2 > 100