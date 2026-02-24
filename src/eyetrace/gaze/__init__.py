"""
Gaze movements module.

This module provides functions to analyze gaze movements including saccades,
fixations, entropy, vergence, and pupil eccentricity.
"""

from .saccades import (
    saccade_velocity,
    saccade_acceleration,
    detect_saccades,
    saccade_fixation_ratio
)
from .fixation import (
    fixation_duration,
    fixation_dispersion,
    gaze_centroid
)
from .entropy import gaze_entropy
from .vergence import vergence_speed
from .eccentricity import pupil_eccentricity
from .vector_3d import gaze_vector_3d

# Try to import Cython-accelerated versions
try:
    from ._saccades_cy import detect_saccades as _detect_saccades_cy
    _HAS_CYTHON_SACC = True
except ImportError:
    _HAS_CYTHON_SACC = False

try:
    from ._fixation_cy import fixation_dispersion as _fixation_dispersion_cy
    _HAS_CYTHON_FIX = True
except ImportError:
    _HAS_CYTHON_FIX = False

try:
    from ._entropy_cy import gaze_entropy as _gaze_entropy_cy
    _HAS_CYTHON_ENT = True
except ImportError:
    _HAS_CYTHON_ENT = False

try:
    from ._vergence_cy import vergence_speed as _vergence_speed_cy
    _HAS_CYTHON_VER = True
except ImportError:
    _HAS_CYTHON_VER = False

# Replace with Cython versions if available
if _HAS_CYTHON_SACC:
    detect_saccades = _detect_saccades_cy
if _HAS_CYTHON_FIX:
    fixation_dispersion = _fixation_dispersion_cy
if _HAS_CYTHON_ENT:
    gaze_entropy = _gaze_entropy_cy
if _HAS_CYTHON_VER:
    vergence_speed = _vergence_speed_cy

__all__ = [
    'saccade_velocity',
    'saccade_acceleration',
    'detect_saccades',
    'saccade_fixation_ratio',
    'fixation_duration',
    'fixation_dispersion',
    'gaze_centroid',
    'gaze_entropy',
    'vergence_speed',
    'pupil_eccentricity',
    'gaze_vector_3d',
]