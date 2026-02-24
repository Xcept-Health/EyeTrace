"""
Tests for eyelids.perclos module.
"""

import pytest
import numpy as np
from eyetrace.eyelids import perclos


def test_perclos_full_window():
    """Test PERCLOS when sequence is shorter than window."""
    ear = np.ones(50) * 0.3
    ear[20:30] = 0.1   # 10 frames closed
    p = perclos(ear, threshold=0.2, window_seconds=10.0, frame_rate=30.0)
    # 50 frames total, 10 closed → 20%
    assert np.isclose(p, 20.0)


def test_perclos_sliding_window():
    """Test PERCLOS with a sliding window (sequence longer than window)."""
    ear = np.ones(100) * 0.3
    ear[20:40] = 0.1   # closed from frame 20 to 39
    # window of 2 seconds = 60 frames at 30 fps
    p = perclos(ear, threshold=0.2, window_seconds=2.0, frame_rate=30.0)
    # The last 60 frames (40-99) have no closure → PERCLOS = 0
    assert np.isclose(p, 0.0)

    # Now make the last 60 frames contain a closure
    ear[70:80] = 0.1   # closed from 70 to 79 (10 frames)
    p = perclos(ear, threshold=0.2, window_seconds=2.0, frame_rate=30.0)
    # In the last 60 frames, 10 closed → 10/60 ≈ 16.667%
    assert np.isclose(p, 100 * 10 / 60, rtol=1e-3)


def test_perclos_no_closure():
    """Test when no frames are below threshold."""
    ear = np.ones(100) * 0.3
    p = perclos(ear, threshold=0.2)
    assert p == 0.0