"""
Unit tests for eyetrace.head_pose.yawning
"""

import pytest
import numpy as np
from eyetrace.head_pose.yawning import yawn_detection, yawn_frequency


def test_yawn_detection():
    """Test yawn detection."""
    mar_seq = np.concatenate([
        np.full(20, 0.35),   # normal
        np.full(40, 0.82),   # long yawn (~1.33s)
        np.full(30, 0.38),
        np.full(10, 0.75),   # too short
        np.full(25, 0.36)
    ])

    yawns = yawn_detection(mar_seq, threshold=0.65, min_duration=0.8, frame_rate=30)
    assert len(yawns) == 1
    start, end = yawns[0]
    assert (end - start + 1) / 30 >= 0.8


def test_yawn_frequency():
    """Test yawn frequency."""
    yawns = [(30, 70), (180, 230)]  # 2 yawns
    freq = yawn_frequency(yawns, total_duration=180.0)  # 3 minutes
    # Verify rate per minute
    assert freq == pytest.approx(0.666, abs=0.01)
