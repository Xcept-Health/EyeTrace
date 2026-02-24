"""
Tests for pupil.metrics module.
"""

import pytest
import numpy as np
from eyetrace.pupil import (
    variance,
    std_dev,
    coefficient_variation,
    normalized_diameter,
    zscore
)

def test_variance():
    """Test variance calculation."""
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    # Unbiased variance = ((1-3)^2 + ... + (5-3)^2) / (5-1) = (4+1+0+1+4)/4 = 10/4 = 2.5
    assert np.isclose(variance(data), 2.5)

    # Single element
    assert variance([42.0]) == 0.0

    # Empty list? Should handle gracefully (maybe return 0 or raise)
    # We'll assume it returns 0
    assert variance([]) == 0.0

def test_std_dev():
    """Test standard deviation."""
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert np.isclose(std_dev(data), np.sqrt(2.5))

    assert std_dev([42.0]) == 0.0
    assert std_dev([]) == 0.0

def test_coefficient_variation():
    """Test coefficient of variation."""
    data = [10, 20, 30, 40, 50]
    # mean=30, std≈15.8114, CV = (15.8114/30)*100 ≈ 52.7047
    assert np.isclose(coefficient_variation(data), 52.7047, rtol=1e-3)

    # Mean zero
    assert coefficient_variation([0, 0, 0]) == 0.0

    # Single element
    assert coefficient_variation([5]) == 0.0

def test_normalized_diameter():
    """Test diameter normalization."""
    data = [2, 4, 6]
    ref = 4
    result = normalized_diameter(data, ref)
    expected = [0.5, 1.0, 1.5]
    np.testing.assert_allclose(result, expected)

    with pytest.raises(ValueError, match="reference_diameter cannot be zero"):
        normalized_diameter(data, 0)

def test_zscore():
    """Test Z-score normalization."""
    data = [1, 2, 3, 4, 5]
    z = zscore(data)
    # mean=3, std≈1.5811, z = [-1.2649, -0.6325, 0, 0.6325, 1.2649]
    expected = [-1.2649, -0.6325, 0, 0.6325, 1.2649]
    np.testing.assert_allclose(z, expected, rtol=1e-3)

    # Constant data
    const = [5, 5, 5]
    np.testing.assert_allclose(zscore(const), [0, 0, 0])

    # Empty array
    assert len(zscore([])) == 0