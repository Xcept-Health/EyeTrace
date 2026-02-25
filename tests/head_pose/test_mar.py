"""
Tests for head_pose.mar module.
"""

import pytest
import numpy as np
from eyetrace.head_pose.mar import mouth_aspect_ratio

# Mock des landmarks MediaPipe
class MockLandmark:
    def __init__(self, x, y, z=0):
        self.x = x
        self.y = y
        self.z = z

class MockFaceLandmarks:
    def __init__(self):
        # Créer suffisamment de landmarks (au moins 300)
        self.landmark = [MockLandmark(0, 0) for _ in range(300)]
        # Définir quelques points spécifiques pour la bouche
        # Indices utilisés : 13 (top), 14 (bottom), 61 (left), 291 (right)
        self.landmark[13] = MockLandmark(0.5, 0.3)
        self.landmark[14] = MockLandmark(0.5, 0.4)
        self.landmark[61] = MockLandmark(0.45, 0.35)
        self.landmark[291] = MockLandmark(0.55, 0.35)

def test_mouth_aspect_ratio():
    """Test MAR calculation."""
    face = MockFaceLandmarks()
    mar = mouth_aspect_ratio(face, 640, 480)
    # Vertical distance: 0.1*480 = 48, horizontal: 0.1*640 = 64, ratio = 48/64 = 0.75
    assert np.isclose(mar, 0.75, rtol=0.1)