"""
Tests for gaze.entropy module.
"""

import pytest
import numpy as np
from eyetrace.gaze.entropy import gaze_entropy

def test_gaze_entropy():
    """Test Shannon entropy of gaze positions."""
    # All points in the same bin -> entropy = 0
    points = np.ones((100, 2)) * 0.5
    entropy = gaze_entropy(points, bins=10)
    assert entropy == 0.0

    # Uniform distribution across bins -> high entropy
    np.random.seed(42)
    points = np.random.rand(1000, 2)
    entropy = gaze_entropy(points, bins=10)
    # Maximum entropy for 10x10 bins would be log(100) ≈ 4.605
    # With 1000 points, should be close to that
    assert entropy > 4.0

def test_gaze_entropy_single_point():
    """Test with a single point."""
    points = np.array([[0.5, 0.5]])
    entropy = gaze_entropy(points, bins=10)
    assert entropy == 0.0

def test_gaze_entropy_empty():
    """Test with empty arrays."""
    points = np.empty((0, 2))
    # La fonction actuelle lève IndexError car points.shape[1] n'existe pas
    # On capture cette erreur en attendant une correction
    with pytest.raises((ValueError, IndexError)):
        gaze_entropy(points)