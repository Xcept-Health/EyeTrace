"""
Tests for pupil.core module.
"""

import pytest
import numpy as np
from eyetrace.pupil import extract_pupil_diameter, extract_iris_radius

def test_extract_pupil_diameter():
    """Test pupil diameter extraction from iris landmarks."""
    # Simulate 5 points on a circle of radius 10
    theta = np.linspace(0, 2*np.pi, 5, endpoint=False)
    radius = 10.0
    landmarks = np.array([
        [radius * np.cos(t) + 50, radius * np.sin(t) + 50] for t in theta
    ], dtype=np.float64)

    diam = extract_pupil_diameter(landmarks, 100, 100)
    # Expected diameter ≈ 2 * mean distance from center
    # Since points are on a perfect circle, mean distance = radius = 10, so diameter = 20
    assert np.isclose(diam, 20.0, rtol=1e-2)

    # Test with conversion to mm
    diam_mm = extract_pupil_diameter(landmarks, 100, 100, px_to_mm=0.05)
    assert np.isclose(diam_mm, 1.0, rtol=1e-2)

def test_extract_iris_radius():
    """Test iris radius extraction."""
    theta = np.linspace(0, 2*np.pi, 5, endpoint=False)
    radius = 12.0
    landmarks = np.array([
        [radius * np.cos(t) + 50, radius * np.sin(t) + 50] for t in theta
    ], dtype=np.float64)

    rad = extract_iris_radius(landmarks, 100, 100)
    assert np.isclose(rad, 12.0, rtol=1e-2)

    rad_mm = extract_iris_radius(landmarks, 100, 100, px_to_mm=0.05)
    assert np.isclose(rad_mm, 0.6, rtol=1e-2)

def test_invalid_shape():
    """Test error on invalid landmark shape."""
    invalid = np.zeros((3, 2))
    with pytest.raises(ValueError, match="Expected (5,2)"):
        extract_pupil_diameter(invalid, 100, 100)

    with pytest.raises(ValueError, match="Expected (5,2)"):
        extract_iris_radius(invalid, 100, 100)

def test_zero_division_handling():
    """Test that function doesn't crash with degenerate landmarks."""
    # All points identical => distances zero
    landmarks = np.ones((5, 2)) * 50
    diam = extract_pupil_diameter(landmarks, 100, 100)
    assert diam == 0.0

    rad = extract_iris_radius(landmarks, 100, 100)
    assert rad == 0.0