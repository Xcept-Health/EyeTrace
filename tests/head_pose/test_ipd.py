"""
Tests for head_pose.ipd module.
"""

import pytest
import numpy as np
from eyetrace.head_pose.ipd import inter_pupillary_distance

def test_inter_pupillary_distance():
    """Test IPD calculation."""
    left = np.array([100, 200])
    right = np.array([150, 200])
    ipd = inter_pupillary_distance(left, right)
    assert np.isclose(ipd, 50.0)