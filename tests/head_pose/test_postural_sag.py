"""
Tests unitaires pour eyetrace.head_pose.postural_sag
"""

import pytest
import numpy as np
from eyetrace.head_pose.postural_sag import postural_sag


def test_postural_sag():
    """Test affaissement postural."""
    times = np.linspace(0, 180, 181)          # 3 minutes
    eye_y = 220.0 + 0.42 * times               # descente progressive
    slope = postural_sag(eye_y, times, baseline_seconds=15)
    assert isinstance(slope, (float, np.floating))
    assert slope > 0.3   # on détecte bien la descente