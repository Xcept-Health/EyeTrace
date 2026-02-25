"""
Tests unitaires pour eyetrace.head_pose.neck_angle
(placeholder en attendant le code complet de neck_angle.py)
"""

import pytest
from eyetrace.head_pose.neck_angle import neck_flexion_angle

# Même mock que pour angles.py
class MockLandmark:
    def __init__(self, x=0.5, y=0.5):
        self.x = float(x)
        self.y = float(y)


class MockFaceLandmarks:
    def __init__(self):
        self.landmark = [MockLandmark() for _ in range(500)]


def test_neck_flexion_angle():
    """Test angle de flexion du cou."""
    face = MockFaceLandmarks()
    angle = neck_flexion_angle(face, 640, 480)

    assert isinstance(angle, (float, np.floating))