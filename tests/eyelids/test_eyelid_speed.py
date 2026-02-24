"""
Tests for eyelids.eyelid_speed module.
"""

import pytest
import numpy as np
from eyetrace.eyelids import eyelid_closing_speed, eyelid_opening_speed


def create_test_ear_signal(fs=100):
    """Create a synthetic EAR signal with one blink."""
    t = np.arange(0, 2, 1/fs)  # 2 seconds
    ear = np.ones_like(t) * 0.3
    # blink from 0.5 to 0.8 seconds
    start = int(0.5 * fs)
    end = int(0.8 * fs)
    # linear closing from 0.3 to 0.1 over 0.15 s, then opening back to 0.3
    closing_len = int(0.15 * fs)
    opening_len = int(0.15 * fs)
    ear[start:start+closing_len] = np.linspace(0.3, 0.1, closing_len)
    ear[start+closing_len:end] = 0.1
    ear[end:end+opening_len] = np.linspace(0.1, 0.3, opening_len)
    return t, ear


def test_eyelid_closing_speed():
    """Test closing speed calculation."""
    t, ear = create_test_ear_signal(fs=100)
    speeds = eyelid_closing_speed(ear, t, threshold=0.2, smooth=False)
    assert len(speeds) == 1
    # The closing speed should be negative, around - (0.2/0.15) ≈ -1.33
    assert speeds[0] < 0
    assert np.isclose(speeds[0], -1.333, rtol=0.1)


def test_eyelid_opening_speed():
    """Test opening speed calculation."""
    t, ear = create_test_ear_signal(fs=100)
    speeds = eyelid_opening_speed(ear, t, threshold=0.2, smooth=False)
    assert len(speeds) == 1
    # Opening speed positive, around 1.33
    assert speeds[0] > 0
    assert np.isclose(speeds[0], 1.333, rtol=0.1)


def test_no_blinks():
    """Test when there are no blinks."""
    t = np.arange(0, 1, 0.01)
    ear = np.ones_like(t) * 0.3
    speeds = eyelid_closing_speed(ear, t)
    assert len(speeds) == 0
    speeds = eyelid_opening_speed(ear, t)
    assert len(speeds) == 0