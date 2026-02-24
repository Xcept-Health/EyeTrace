"""
Tests for gaze.eccentricity module.
"""

import pytest
import numpy as np
from eyetrace.gaze.eccentricity import pupil_eccentricity

def test_pupil_eccentricity():
    """Test pupil eccentricity (angle correction)."""
    pupil_center = np.array([100, 100])
    iris_center = np.array([102, 100])
    gaze_vector = np.array([0, 0, 1])  # vecteur de regard factice
    eccentricity = pupil_eccentricity(pupil_center, iris_center, gaze_vector)
    assert isinstance(eccentricity, float)
    assert np.isfinite(eccentricity)