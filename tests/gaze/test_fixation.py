"""
Tests for gaze.fixation module.
"""

import pytest
import numpy as np
from eyetrace.gaze.fixation import (
    fixation_duration,
    fixation_dispersion,
    gaze_centroid
)

def test_fixation_duration():
    """Test computation of fixation durations."""
    labels = np.array([1, 1, 1, 0, 0, 1, 1, 1, 1])
    t = np.arange(len(labels))
    durations = fixation_duration(labels, t)
    # Fixation regions: indices 0-2 (duration = t[2]-t[0] = 2) and 5-8 (duration = t[8]-t[5] = 3)
    expected = [2.0, 3.0]
    np.testing.assert_allclose(durations, expected)

def test_fixation_dispersion():
    """Test dispersion (standard deviation) during a fixation."""
    # Create a 2D array of positions (x, y)
    points = np.array([
        [10, 20],
        [11, 20.1],
        [10.5, 19.9],
        [10.2, 20.0],
        [10.8, 20.2]
    ])
    # Fixation mask (all True for this example)
    mask = np.ones(len(points), dtype=bool)
    disp = fixation_dispersion(points, mask)
    # The function returns a list of dispersions (one per fixation)
    assert isinstance(disp, list)
    assert len(disp) == 1
    assert disp[0] > 0

def test_gaze_centroid():
    """Test centroid (mean position) of gaze points."""
    points = np.array([
        [1, 4],
        [2, 5],
        [3, 6]
    ])
    centroid = gaze_centroid(points)
    np.testing.assert_allclose(centroid, [2, 5])
