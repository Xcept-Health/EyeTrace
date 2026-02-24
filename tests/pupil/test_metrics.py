import pytest
import numpy as np
from eyetrace.pupil import (
    variance, std_dev, coefficient_variation,
    normalized_diameter, zscore
)

def test_variance():
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert np.isclose(variance(data), 2.5)
    assert variance([42.0]) == 0.0

def test_std_dev():
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert np.isclose(std_dev(data), np.sqrt(2.5))

def test_coefficient_variation():
    data = [10, 20, 30, 40, 50]
    assert np.isclose(coefficient_variation(data), 52.7047, rtol=1e-3)
    assert coefficient_variation([0, 0, 0]) == 0.0

def test_normalized_diameter():
    data = [2, 4, 6]
    result = normalized_diameter(data, 4)
    np.testing.assert_allclose(result, [0.5, 1.0, 1.5])
    
    with pytest.raises(ValueError):
        normalized_diameter(data, 0)

def test_zscore():
    data = [1, 2, 3, 4, 5]
    z = zscore(data)
    expected = [-1.2649, -0.6325, 0, 0.6325, 1.2649]
    np.testing.assert_allclose(z, expected, rtol=1e-3)
    
    # Données constantes
    const = [5, 5, 5]
    np.testing.assert_allclose(zscore(const), [0, 0, 0])