"""
Tests for head_pose.postural_sag module.
"""

import pytest
import numpy as np
from eyetrace.head_pose.postural_sag import postural_sag

def test_postural_sag():
    """Test postural sag (trend of eye y-coordinate)."""
    t = np.linspace(0, 10, 100)
    y = 200 + 0.5 * t  # pente positive = descente
    slope = postural_sag(y, t, baseline_seconds=2)
    assert np.isclose(slope, 0.5, rtol=0.1)