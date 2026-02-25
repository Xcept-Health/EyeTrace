"""
Tests for eyetrace.head_pose.postural_sag
"""

import pytest
import numpy as np
from eyetrace.head_pose.postural_sag import postural_sag


def test_postural_sag():
    """Test postural sag."""
    times = np.linspace(0, 180, 181)          # 3 minutes
    eye_y = 220.0 + 0.42 * times               # gradual downward shift
    slope = postural_sag(eye_y, times, baseline_seconds=15)
    assert isinstance(slope, (float, np.floating))
    assert slope > 0.3   # downward trend is correctly detected
