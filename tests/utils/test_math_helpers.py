"""
Tests for utils.math_helpers module.
"""

import pytest
import numpy as np
from eyetrace.utils.math_helpers import (
    mad_outlier_removal,
    sliding_window_view,
    find_peaks
)

def test_mad_outlier_removal():
    """Test MAD-based outlier removal."""
    arr = np.array([1, 2, 3, 4, 100])
    mask, cleaned = mad_outlier_removal(arr, threshold=3.5)
    # The last point (100) should be identified as an outlier
    assert not mask[4]
    assert np.isnan(cleaned[4])

def test_sliding_window_view():
    """Test sliding window view."""
    arr = np.arange(10)
    windows = sliding_window_view(arr, window_size=3)
    # Check window shape and verify first/last window contents
    assert windows.shape == (8, 3)
    np.testing.assert_array_equal(windows[0], [0, 1, 2])
    np.testing.assert_array_equal(windows[-1], [7, 8, 9])

def test_find_peaks():
    """Test simple peak detection."""
    arr = np.array([1, 3, 1, 2, 5, 2, 1])
    peaks = find_peaks(arr, height=2, distance=2)
    # Peak indices should match expected local maxima
    expected = np.array([1, 4])
    np.testing.assert_array_equal(peaks, expected)
