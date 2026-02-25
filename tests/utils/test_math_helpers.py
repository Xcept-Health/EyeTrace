"""
Tests for utils.math_helpers module.
"""

import pytest
import numpy as np
from eyetrace.utils.math_helpers import (
    normalize,
    zscore,
    mad,
    running_mean,
    nan_remove
)

def test_normalize():
    """Test vector normalization."""
    v = np.array([3, 4])
    norm = normalize(v)
    np.testing.assert_allclose(norm, [0.6, 0.8])

def test_zscore():
    """Test Z-score normalization of array."""
    arr = np.array([1, 2, 3, 4, 5])
    z = zscore(arr)
    expected = (arr - np.mean(arr)) / np.std(arr, ddof=1)
    np.testing.assert_allclose(z, expected)

def test_mad():
    """Test median absolute deviation."""
    arr = np.array([1, 2, 3, 4, 100])
    mad_val = mad(arr)
    # MAD sans outlier doit être ~1.48 * median(|x_i - median|)
    # On teste juste que c'est un float
    assert isinstance(mad_val, float)

def test_running_mean():
    """Test running mean."""
    arr = np.arange(10)
    rm = running_mean(arr, window=3)
    assert len(rm) == len(arr)

def test_nan_remove():
    """Test removal of NaN values from array."""
    arr = np.array([1, 2, np.nan, 4, 5])
    cleaned = nan_remove(arr)
    assert not np.any(np.isnan(cleaned))
    assert len(cleaned) == 4