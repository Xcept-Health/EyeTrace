"""
Tests for head_pose.utils module.
"""

import pytest
import numpy as np
from eyetrace.head_pose.utils import (
    get_face_landmark_array,
    FACE_MODEL_POINTS,
    FACE_MODEL_INDICES
)

class MockLandmark:
    def __init__(self, x, y, z=0):
        self.x = x
        self.y = y
        self.z = z

class MockFaceLandmarks:
    def __init__(self):
        self.landmark = [MockLandmark(i/100, i/100) for i in range(500)]

def test_get_face_landmark_array():
    """Test extraction of landmark array."""
    face = MockFaceLandmarks()
    indices = [1, 2, 3]
    arr = get_face_landmark_array(face, indices, 640, 480)
    assert arr.shape == (3, 2)
    assert np.all(arr >= 0)

def test_face_model_constants():
    """Test that model constants are defined."""
    assert FACE_MODEL_POINTS.shape == (6, 3)
    assert len(FACE_MODEL_INDICES) == 6