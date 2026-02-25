"""
Tests unitaires pour eyetrace.head_pose.angles
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # EyeTrace/

import pytest
import numpy as np
from eyetrace.head_pose.angles import head_pose_angles

# ====================== MOCKS ======================
class MockLandmark:
    def __init__(self, x=0.5, y=0.5, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


class MockFaceLandmarks:
    def __init__(self):
        self.landmark = [MockLandmark() for _ in range(500)]
        # Points utilisés par solvePnP
        self.landmark[1]   = MockLandmark(0.50, 0.48)   # nose tip
        self.landmark[152] = MockLandmark(0.50, 0.68)   # chin
        self.landmark[33]  = MockLandmark(0.38, 0.42)   # left eye left
        self.landmark[263] = MockLandmark(0.62, 0.42)   # right eye right
        self.landmark[61]  = MockLandmark(0.43, 0.57)   # left mouth
        self.landmark[291] = MockLandmark(0.57, 0.57)   # right mouth


def test_head_pose_angles():
    """Test estimation des angles de tête (pitch, roll, yaw)."""
    face = MockFaceLandmarks()
    pitch, roll, yaw = head_pose_angles(face, 640, 480)

    assert isinstance(pitch, (float, np.floating))
    assert isinstance(roll, (float, np.floating))
    assert isinstance(yaw, (float, np.floating))