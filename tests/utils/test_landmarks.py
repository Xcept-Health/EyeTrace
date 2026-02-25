"""
Tests for utils.landmarks module.
"""

import pytest
import numpy as np
from eyetrace.utils.landmarks import (
    extract_eye_landmarks,
    extract_iris_landmarks,
    extract_face_landmarks
)

class MockLandmark:
    def __init__(self, x, y, z=0):
        self.x = x
        self.y = y
        self.z = z

class MockFaceLandmarks:
    def __init__(self):
        self.landmark = [MockLandmark(i/100, i/100) for i in range(500)]

def test_extract_eye_landmarks():
    """Test extraction of eye landmarks."""
    face = MockFaceLandmarks()
    left, right = extract_eye_landmarks(face, 640, 480)
    assert left.shape == (6, 2)
    assert right.shape == (6, 2)

def test_extract_iris_landmarks():
    """Test extraction of iris landmarks."""
    face = MockFaceLandmarks()
    left_iris = extract_iris_landmarks(face, 640, 480, eye='left')
    right_iris = extract_iris_landmarks(face, 640, 480, eye='right')
    assert left_iris.shape == (5, 2)
    assert right_iris.shape == (5, 2)

def test_extract_face_landmarks():
    """Test extraction of face landmarks as array."""
    face = MockFaceLandmarks()
    arr = extract_face_landmarks(face, 640, 480)
    assert arr.shape[1] == 2
    assert arr.shape[0] > 0