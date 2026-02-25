"""
Tests for head_pose.yawning module.
"""

import pytest
import numpy as np
from eyetrace.head_pose.yawning import yawn_detection, yawn_frequency

def test_yawn_detection():
    """Test yawn detection from MAR sequence."""
    mar = np.ones(300) * 0.3
    # Simuler un bâillement long (200 frames > threshold 0.6)
    mar[100:300] = 0.8
    yawns = yawn_detection(mar, threshold=0.6, min_duration=2.0, frame_rate=30)
    # 200 frames = 6.67 sec, donc détecté
    assert len(yawns) == 1
    assert yawns[0][0] == 100
    assert yawns[0][1] == 299

def test_yawn_frequency():
    """Test yawn frequency calculation."""
    yawns = [(10, 20), (30, 40)]
    freq = yawn_frequency(yawns, total_duration=60.0)
    assert np.isclose(freq, 2.0)  # 2 yawns per minute
    freq = yawn_frequency([], 60.0)
    assert freq == 0.0