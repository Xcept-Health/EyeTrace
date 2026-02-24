"""
Tests for eyelids.ear module.
"""

import pytest
import numpy as np
from eyetrace.eyelids import eye_aspect_ratio

# Optionally test Cython version if available
try:
    from eyetrace.eyelids._ear_cy import eye_aspect_ratio as cy_ear
    HAS_CYTHON_EAR = True
except ImportError:
    HAS_CYTHON_EAR = False


def test_ear_normal():
    """Test EAR calculation with a normal open eye."""
    eye = np.array([
        [30, 30],  # p1: outer corner
        [32, 25],  # p2: top outer
        [38, 25],  # p3: top inner
        [40, 30],  # p4: inner corner
        [38, 35],  # p5: bottom inner
        [32, 35]   # p6: bottom outer
    ], dtype=np.float64)

    ear = eye_aspect_ratio(eye)
    # Expected EAR: (vertical1 + vertical2) / (2 * horizontal)
    # vertical1 ≈ 10, vertical2 ≈ 10, horizontal ≈ 10 => EAR = (10+10)/(2*10)=1.0
    assert np.isclose(ear, 1.0, rtol=0.1)


def test_ear_closed():
    """Test EAR with a closed eye (vertical distances small)."""
    eye = np.array([
        [30, 30],
        [32, 30],
        [38, 30],
        [40, 30],
        [38, 30],
        [32, 30]
    ], dtype=np.float64)

    ear = eye_aspect_ratio(eye)
    assert ear < 0.1


def test_ear_zero_horizontal():
    """Test when horizontal distance is zero (should return 0)."""
    eye = np.array([
        [30, 30],
        [32, 25],
        [38, 25],
        [30, 30],   # same as p1 => horizontal=0
        [38, 35],
        [32, 35]
    ], dtype=np.float64)

    ear = eye_aspect_ratio(eye)
    assert ear == 0.0


def test_ear_invalid_shape():
    """Test error on invalid input shape."""
    with pytest.raises(ValueError, match="Expected (6, 2)"):
        eye_aspect_ratio(np.zeros((5, 2)))


@pytest.mark.skipif(not HAS_CYTHON_EAR, reason="Cython version not compiled")
def test_ear_cython_consistency():
    """Test that Cython and Python versions give the same result."""
    eye = np.array([
        [30, 30],
        [32, 25],
        [38, 25],
        [40, 30],
        [38, 35],
        [32, 35]
    ], dtype=np.float64)

    py_result = eye_aspect_ratio(eye)  # This might be the Python version if not replaced
    # To be safe, import Python version directly
    from eyetrace.eyelids.ear import eye_aspect_ratio as py_ear
    py_val = py_ear(eye)
    cy_val = cy_ear(eye)
    assert np.isclose(py_val, cy_val)
    # Also check that the public API returns the same
    assert np.isclose(eye_aspect_ratio(eye), cy_val)