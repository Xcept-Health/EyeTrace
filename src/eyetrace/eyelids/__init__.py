"""
Eyelid and blink analysis module.

This module provides functions to extract eyelid metrics such as EAR,
blink detection, PERCLOS, eyelid speeds, symmetry, and microsleep indicators.
"""

# Import public API from submodules
from .ear import eye_aspect_ratio
from .blink_detection import (
    detect_blinks,
    blink_frequency,
    mean_closure_duration,
    long_blink_ratio
)
from .perclos import perclos
from .eyelid_speed import (
    eyelid_closing_speed,
    eyelid_opening_speed
)
from .symmetry import eyelid_symmetry
from .jerk import ear_jerk
from .microsleep import microsleep_indicator

# Try to import Cython-accelerated versions
try:
    from ._ear_cy import eye_aspect_ratio as _ear_cy
    _HAS_CYTHON_EAR = True
except ImportError:
    _HAS_CYTHON_EAR = False

# Replace with Cython version if available
if _HAS_CYTHON_EAR:
    eye_aspect_ratio = _ear_cy

__all__ = [
    'eye_aspect_ratio',
    'detect_blinks',
    'blink_frequency',
    'mean_closure_duration',
    'long_blink_ratio',
    'perclos',
    'eyelid_closing_speed',
    'eyelid_opening_speed',
    'eyelid_symmetry',
    'ear_jerk',
    'microsleep_indicator',
]