"""
Unit tests for eyetrace.head_pose.neck_angle
"""

import pytest
import numpy as np
from unittest.mock import patch
from eyetrace.head_pose.neck_angle import neck_flexion_angle

class MockLandmark:
    def __init__(self, x=0.5, y=0.5):
        self.x = float(x)
        self.y = float(y)

class MockFaceLandmarks:
    def __init__(self):
        # Create a list of 500 mock landmarks
        self.landmark = [MockLandmark() for _ in range(500)]

@patch('eyetrace.head_pose.angles.head_pose_angles')
def test_neck_flexion_angle(mock_head_pose):
    """Test neck flexion angle calculation."""
    # Simulate head_pose_angles return value: pitch, roll, yaw
    mock_head_pose.return_value = (0.1, 0.0, 0.0)

    face = MockFaceLandmarks()
    angle = neck_flexion_angle(face, 640, 480)

    assert isinstance(angle, float)
    assert np.isclose(angle, 0.1)
    mock_head_pose.assert_called_once_with(face, 640, 480, camera_matrix=None)
