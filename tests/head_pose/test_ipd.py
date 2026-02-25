"""
Tests unitaires pour eyetrace.head_pose.ipd
"""

import pytest
import numpy as np
from eyetrace.head_pose.ipd import inter_pupillary_distance


def test_inter_pupillary_distance():
    """Test distance inter-pupillaire."""
    left = np.array([310.0, 240.0])
    right = np.array([370.0, 240.0])
    ipd = inter_pupillary_distance(left, right)
    assert isinstance(ipd, float)
    assert ipd == pytest.approx(60.0, abs=0.01)