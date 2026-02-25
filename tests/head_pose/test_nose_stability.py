"""
Tests unitaires pour eyetrace.head_pose.nose_stability
"""

import pytest
import numpy as np
from eyetrace.head_pose.nose_stability import nose_stability

# ====================== MOCKS ======================
class MockLandmark:
    def __init__(self, x=0.5, y=0.5):
        self.x = float(x)
        self.y = float(y)


class MockFaceLandmarks:
    def __init__(self):
        self.landmark = [MockLandmark() for _ in range(500)]
        self.landmark[1] = MockLandmark(0.50, 0.48)  # nose tip


def create_mock_sequence(n=50):
    seq = [MockFaceLandmarks() for _ in range(n)]
    for i, face in enumerate(seq):
        face.landmark[1].x = 0.50 + (i - 25) * 0.002   # léger mouvement
    return seq


def test_nose_stability():
    """Test variance de la position du nez."""
    seq = create_mock_sequence(60)
    var = nose_stability(seq, 640, 480)
    assert isinstance(var, (float, np.floating))
    assert var >= 0.0