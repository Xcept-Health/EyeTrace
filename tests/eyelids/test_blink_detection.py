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


# Tests for detect_blinks

def test_detect_blinks_basic():
    """Test blink detection on a simple EAR signal with two blinks."""
    ear = np.ones(100) * 0.3
    ear[20:25] = 0.1   # first blink, 5 frames
    ear[60:63] = 0.1   # second blink, 3 frames

    blinks = detect_blinks(ear, threshold=0.2, min_interval_frames=10)
    assert len(blinks) == 2
    assert blinks[0] == (20, 24)  # indices start/end inclus
    assert blinks[1] == (60, 62)


def test_detect_blinks_merge_when_close():
    """Test merging of blinks that are too close."""
    ear = np.ones(100) * 0.3
    ear[20:25] = 0.1
    ear[28:30] = 0.1   # only 3 frames apart
    blinks = detect_blinks(ear, threshold=0.2, min_interval_frames=5)
    # Should merge into one blink from 20 to 29
    assert len(blinks) == 1
    assert blinks[0] == (20, 29)


def test_detect_blinks_no_merge_when_far():
    """Test that blinks sufficiently apart are not merged."""
    ear = np.ones(100) * 0.3
    ear[20:25] = 0.1
    ear[40:45] = 0.1   # 15 frames apart ( > min_interval_frames=10)
    blinks = detect_blinks(ear, threshold=0.2, min_interval_frames=10)
    assert len(blinks) == 2
    assert blinks[0] == (20, 24)
    assert blinks[1] == (40, 44)


def test_detect_blinks_no_blinks():
    """Test with no blinks (all EAR above threshold)."""
    ear = np.ones(100) * 0.3
    blinks = detect_blinks(ear, threshold=0.2)
    assert len(blinks) == 0


def test_detect_blinks_single_frame_region():
    """Test that a single frame below threshold is not counted as a blink."""
    ear = np.ones(100) * 0.3
    ear[50] = 0.1   # single frame below threshold
    blinks = detect_blinks(ear, threshold=0.2)
    assert len(blinks) == 0


def test_detect_blinks_empty_array():
    """Test with an empty EAR array."""
    ear = np.array([])
    blinks = detect_blinks(ear, threshold=0.2)
    assert blinks == []


def test_detect_blinks_threshold_too_low():
    """Test that no blinks are detected if threshold is below all values."""
    ear = np.ones(100) * 0.3
    ear[20:25] = 0.1
    blinks = detect_blinks(ear, threshold=0.05)  # below 0.1
    assert len(blinks) == 0


def test_detect_blinks_all_below_threshold():
    """Test when the whole sequence is below threshold (constant closure)."""
    ear = np.ones(100) * 0.1
    blinks = detect_blinks(ear, threshold=0.2)
    # Should detect one long blink
    assert len(blinks) == 1
    assert blinks[0] == (0, 99)


def test_detect_blinks_boundary_at_edges():
    """Test blinks that start at the very beginning or end."""
    ear = np.ones(100) * 0.3
    ear[0:5] = 0.1      # blink at start
    ear[95:100] = 0.1   # blink at end
    blinks = detect_blinks(ear, threshold=0.2, min_interval_frames=10)
    assert len(blinks) == 2
    assert blinks[0] == (0, 4)
    assert blinks[1] == (95, 99)



# Tests for blink_frequency

def test_blink_frequency_normal():
    """Test blink frequency with a list of blinks."""
    blinks = [(10, 12), (30, 32), (50, 52)]
    freq = blink_frequency(blinks, duration_seconds=60.0)
    assert np.isclose(freq, 3.0)  # 3 blinks per minute

    freq = blink_frequency([], 60.0)
    assert freq == 0.0


def test_blink_frequency_zero_duration():
    """Test blink frequency with zero duration."""
    blinks = [(10, 12)]
    freq = blink_frequency(blinks, duration_seconds=0.0)
    assert freq == 0.0



# Tests for mean_closure_duration

def test_mean_closure_duration_normal():
    """Test mean closure duration with several blinks."""
    blinks = [(10, 12), (30, 33)]  # 3 and 4 frames
    mcd = mean_closure_duration(blinks, frame_rate=30.0)
    # (3+4)/2 = 3.5 frames = 3.5/30 ≈ 0.1166667 s
    assert np.isclose(mcd, 3.5 / 30.0)

    mcd = mean_closure_duration([], 30.0)
    assert mcd == 0.0


def test_mean_closure_duration_single_blink():
    """Test with a single blink."""
    blinks = [(10, 12)]  # 3 frames
    mcd = mean_closure_duration(blinks, frame_rate=30.0)
    assert np.isclose(mcd, 3.0 / 30.0)



# Tests for long_blink_ratio

def test_long_blink_ratio_mixed():
    """Test ratio with a mix of short and long blinks."""
    blinks = [(10, 12), (30, 35), (50, 60)]   # durations: 3, 6, 11 frames at 30 fps → 0.1, 0.2, 0.3667 s
    ratio = long_blink_ratio(blinks, frame_rate=30.0, threshold_seconds=0.19)
    # blinks > 0.19s: indices 1 and 2 → 2/3 ≈ 0.6667
    assert np.isclose(ratio, 2.0 / 3.0)

    ratio = long_blink_ratio(blinks, frame_rate=30.0, threshold_seconds=0.5)
    assert ratio == 0.0


def test_long_blink_ratio_all_short():
    """Test when all blinks are shorter than threshold."""
    blinks = [(10, 12), (30, 32)]  # 3 frames each → 0.1 s
    ratio = long_blink_ratio(blinks, frame_rate=30.0, threshold_seconds=0.5)
    assert ratio == 0.0


def test_long_blink_ratio_all_long():
    """Test when all blinks are longer than threshold."""
    blinks = [(10, 30), (40, 60)]  # 21 frames each → 0.7 s
    ratio = long_blink_ratio(blinks, frame_rate=30.0, threshold_seconds=0.5)
    assert ratio == 1.0


def test_long_blink_ratio_empty():
    """Test with empty blink list."""
    ratio = long_blink_ratio([], frame_rate=30.0, threshold_seconds=0.5)
    assert ratio == 0.0


def test_long_blink_ratio_exact_threshold():
    """Test behavior when duration equals threshold exactly."""
    # Créer un blink dont la durée est exactement threshold_seconds
    blinks = [(10, 19)]  # 10 frames à 30 fps = 10/30 ≈ 0.3333 s
    ratio = long_blink_ratio(blinks, frame_rate=30.0, threshold_seconds=10/30.0)
    # Selon l'implémentation, > ou >= ? Ici on utilise > d'après le code.
    # Si durée == threshold, alors d > threshold est False, donc pas compté.
    assert ratio == 0.0