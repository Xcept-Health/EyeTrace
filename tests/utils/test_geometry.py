"""
Tests for utils.geometry module.
"""

import pytest
import numpy as np
from eyetrace.utils.geometry import (
    angle_between_vectors,
    distance,
    normalize_vector,
    project_point_to_line
)

def test_angle_between_vectors():
    """Test angle calculation between vectors."""
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])
    angle = angle_between_vectors(v1, v2, in_degrees=True)
    assert np.isclose(angle, 90.0)

def test_distance():
    """Test Euclidean distance."""
    p1 = np.array([0, 0])
    p2 = np.array([3, 4])
    d = distance(p1, p2)
    assert np.isclose(d, 5.0)

def test_normalize_vector():
    """Test vector normalization."""
    v = np.array([3, 4])
    n = normalize_vector(v)
    assert np.isclose(np.linalg.norm(n), 1.0)

def test_project_point_to_line():
    """Test projection of point onto line."""
    p = np.array([1, 1])
    a = np.array([0, 0])
    b = np.array([2, 0])
    proj = project_point_to_line(p, a, b)
    np.testing.assert_allclose(proj, [1, 0])