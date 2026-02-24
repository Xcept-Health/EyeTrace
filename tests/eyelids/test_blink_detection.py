"""
Tests for eyelids.blink_detection module.
"""

import pytest
import numpy as np
from eyetrace.eyelids import (
    detect_blinks,
    blink_frequency,
    mean_closure_duration,
    long_blink_ratio
)


def test_detect_blinks():
    """Test blink detection on a simple EAR signal."""
    # Create an EAR signal with two blinks
    ear = np.ones(100) * 0.3
    ear[20:25] = 0.1   # first blink, 5 frames
    ear[60:63] = 0.1   # second blink, 3 frames

    blinks = detect_blinks(ear, threshold=0.2, min_interval_frames=10)
    assert len(blinks) == 2
    assert blinks[0] == (20, 24)  # (start, end)
    assert blinks[1] == (60, 62)


def test_detect_blinks_merge():
    """Test merging of blinks that are too close."""
    ear = np.ones(100) * 0.3
    ear[20:25] = 0.1
    ear[28:30] = 0.1   # only 3 frames apart
    blinks = detect_blinks(ear, threshold=0.2, min_interval_frames=5)
    # Should merge into one blink from 20 to 29
    assert len(blinks) == 1
    assert blinks[0] == (20, 29)


def test_detect_blinks_no_blinks():
    """Test with no blinks (all EAR above threshold)."""
    ear = np.ones(100) * 0.3
    blinks = detect_blinks(ear, threshold=0.2)
    assert len(blinks) == 0


def test_blink_frequency():
    """Test blink frequency calculation."""
    blinks = [(10, 12), (30, 32), (50, 52)]
    freq = blink_frequency(blinks, duration_seconds=60.0)
    assert np.isclose(freq, 3.0)  # 3 blinks per minute

    freq = blink_frequency([], 60.0)
    assert freq == 0.0


def test_mean_closure_duration():
    """Test mean closure duration."""
    blinks = [(10, 12), (30, 33)]  # 3 and 4 frames
    mcd = mean_closure_duration(blinks, frame_rate=30.0)
    # (3+4)/2 = 3.5 frames = 3.5/30 ≈ 0.1167 s
    assert np.isclose(mcd, 0.1167, rtol=1e-3)

    mcd = mean_closure_duration([], 30.0)
    assert mcd == 0.0


def test_long_blink_ratio():
    """Test ratio of long blinks."""
    blinks = [(10, 12), (30, 35), (50, 60)]
    # durations: (3, 6, 11) frames at 30 fps → 0.1, 0.2, 0.3667 s
    # Utiliser un seuil de 0.19 pour que les deux derniers soient comptés
    ratio = long_blink_ratio(blinks, frame_rate=30.0, threshold_seconds=0.19)
    # blinks > 0.19s : indices 1 et 2 → 2/3 ≈ 0.667
    assert np.isclose(ratio, 2/3)