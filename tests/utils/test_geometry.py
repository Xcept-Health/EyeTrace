"""
Tests for utils.geometry module.
"""

import pytest
import numpy as np
from eyetrace.utils.geometry import (
    angle_between_vectors,
    distance_point_to_line,
    circle_fit,
    rotate_point
)

def test_angle_between_vectors():
    """Test angle calculation between vectors."""
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])
    angle = angle_between_vectors(v1, v2)
    assert np.isclose(angle, np.pi/2)

def test_distance_point_to_line():
    """Test distance from point to line."""
    p = np.array([1, 1])
    a = np.array([0, 0])
    b = np.array([1, 0])
    dist = distance_point_to_line(p, a, b)
    assert np.isclose(dist, 1.0)

def test_circle_fit():
    """Test circle fitting to points."""
    points = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    center, radius = circle_fit(points)
    np.testing.assert_allclose(center, [0, 0], atol=0.1)
    assert np.isclose(radius, 1.0, rtol=0.1)

def test_rotate_point():
    """Test 2D point rotation."""
    p = np.array([1, 0])
    rotated = rotate_point(p, np.pi/2, origin=[0, 0])
    np.testing.assert_allclose(rotated, [0, 1], atol=1e-6)