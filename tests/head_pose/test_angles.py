"""
Tests for head_pose.angles module.
"""

import pytest
import numpy as np
from eyetrace.head_pose.angles import head_pose_angles

# Mock des landmarks MediaPipe pour les tests
class MockLandmark:
    def __init__(self, x, y, z=0):
        self.x = x
        self.y = y
        self.z = z

class MockFaceLandmarks:
    def __init__(self):
        # Créer 500 landmarks factices (MediaPipe en a 468)
        self.landmark = [MockLandmark(i/100, i/100) for i in range(500)]

def test_head_pose_angles():
    """Test head pose angles estimation."""
    face = MockFaceLandmarks()
    pitch, roll, yaw = head_pose_angles(face, 640, 480)
    # Les angles doivent être des floats (peuvent être nan si échec)
    assert isinstance(pitch, float)
    assert isinstance(roll, float)
    assert isinstance(yaw, float)