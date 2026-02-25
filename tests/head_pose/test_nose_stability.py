"""
Tests for head_pose.nose_stability module.
"""

import pytest
import numpy as np
from eyetrace.head_pose.nose_stability import nose_stability

class MockLandmark:
    def __init__(self, x, y, z=0):
        self.x = x
        self.y = y
        self.z = z

class MockFaceLandmarks:
    def __init__(self, x, y):
        self.landmark = [MockLandmark(0, 0) for _ in range(10)]
        # index 1 = nose tip
        self.landmark[1] = MockLandmark(x, y)

def test_nose_stability():
    """Test nose stability (variance of nose position)."""
    # Simuler une séquence de landmarks avec un mouvement linéaire
    seq = []
    for i in range(10):
        face = MockFaceLandmarks(0.5 + i*0.01, 0.5)
        seq.append(face)
    var = nose_stability(seq, 640, 480)
    # La variance en x devrait être non nulle
    assert var > 0
    # Avec mouvement constant, variance = np.var(np.linspace(0, 0.09, 10)*640) ≈ ?
    # On vérifie simplement que c'est un float.
    assert isinstance(var, float)