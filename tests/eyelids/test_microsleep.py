"""
Tests for eyelids.microsleep module.
"""

import pytest
import numpy as np
from eyetrace.eyelids import microsleep_indicator


def test_microsleep_indicator():
    """Test detection of microsleep events."""
    ear = np.ones(300) * 0.3
    # Insert a long closure from frame 100 to 199 (100 frames)
    ear[100:200] = 0.1
    indicator = microsleep_indicator(ear, frame_rate=30.0,
                                      ear_threshold=0.2,
                                      duration_threshold=2.0)
    # 100 frames at 30 fps = 3.33 seconds, so should be detected
    assert np.any(indicator)
    # The indicator should be True exactly where the closure is
    assert np.all(indicator[100:200])
    assert not np.any(indicator[:100])
    assert not np.any(indicator[200:])


def test_microsleep_indicator_short_closure():
    """Test that a short closure is not considered microsleep."""
    ear = np.ones(300) * 0.3
    ear[100:120] = 0.1   # 20 frames = 0.67 s
    indicator = microsleep_indicator(ear, frame_rate=30.0,
                                      ear_threshold=0.2,
                                      duration_threshold=2.0)
    assert not np.any(indicator)


def test_microsleep_indicator_no_closure():
    """Test with no closure."""
    ear = np.ones(300) * 0.3
    indicator = microsleep_indicator(ear, frame_rate=30.0)
    assert not np.any(indicator)