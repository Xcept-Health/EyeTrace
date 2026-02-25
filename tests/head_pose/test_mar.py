"""
Tests unitaires pour eyetrace.head_pose.mar
"""

import pytest
import numpy as np
from eyetrace.head_pose.mar import mouth_aspect_ratio

# ====================== MOCKS ======================
class MockLandmark:
    def __init__(self, x=0.5, y=0.5):
        self.x = float(x)
        self.y = float(y)


class MockFaceLandmarks:
    def __init__(self):
        self.landmark = [MockLandmark() for _ in range(500)]
        self.landmark[13]  = MockLandmark(0.50, 0.53)   # upper lip
        self.landmark[14]  = MockLandmark(0.50, 0.59)   # lower lip
        self.landmark[61]  = MockLandmark(0.43, 0.56)   # left corner
        self.landmark[291] = MockLandmark(0.57, 0.56)   # right corner

    def set_mouth_open(self, intensity=0.15):
        self.landmark[13].y = 0.52 - intensity
        self.landmark[14].y = 0.58 + intensity


def test_mouth_aspect_ratio_closed():
    """Test MAR bouche fermée."""
    face = MockFaceLandmarks()
    mar = mouth_aspect_ratio(face, 640, 480)
    assert isinstance(mar, float)
    assert 0.0 <= mar < 0.6


def test_mouth_aspect_ratio_open():
    """Test MAR bouche ouverte."""
    face = MockFaceLandmarks()
    face.set_mouth_open(0.18)
    mar = mouth_aspect_ratio(face, 640, 480)
    assert mar > 0.65