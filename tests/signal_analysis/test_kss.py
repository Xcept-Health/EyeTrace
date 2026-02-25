"""
Tests for signal_analysis.kss module.
"""

import pytest
import numpy as np
from eyetrace.signal_analysis.kss import kss_prediction

def test_kss_prediction():
    """Test KSS prediction (placeholder)."""
    # À adapter selon l'implémentation réelle
    features = np.random.randn(10)
    kss = kss_prediction(features)
    assert 1 <= kss <= 9 or np.isnan(kss)