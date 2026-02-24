"""
Tests for eyelids.symmetry module.
"""

import pytest
import numpy as np
from eyetrace.eyelids import eyelid_symmetry


def test_eyelid_symmetry_perfect():
    """Test symmetry when left and right are identical."""
    left = np.array([0.3, 0.3, 0.2, 0.2, 0.3])
    right = np.array([0.3, 0.3, 0.2, 0.2, 0.3])
    corr = eyelid_symmetry(left, right)
    assert np.isclose(corr, 1.0)


def test_eyelid_symmetry_opposite():
    """Test negative correlation."""
    left = np.array([0.3, 0.3, 0.2, 0.2, 0.3])
    right = 1.0 - left  # inverted
    corr = eyelid_symmetry(left, right)
    assert np.isclose(corr, -1.0, rtol=1e-3)


def test_eyelid_symmetry_random():
    """Test with random data (should give some value)."""
    np.random.seed(42)
    left = np.random.randn(100)
    right = np.random.randn(100)
    corr = eyelid_symmetry(left, right)
    # Just check it's a float between -1 and 1
    assert -1 <= corr <= 1


def test_eyelid_symmetry_insufficient_data():
    """Test with less than 2 points."""
    left = np.array([0.3])
    right = np.array([0.3])
    corr = eyelid_symmetry(left, right)
    assert np.isnan(corr)