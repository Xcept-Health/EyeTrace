"""
Pupil dynamics module.

This module provides functions to extract and analyze pupil metrics from
eye tracking data. It includes both pure Python implementations and
optimized Cython versions for performance-critical functions.
"""

# Import public API
from .core import extract_pupil_diameter, extract_iris_radius
from .metrics import (
    variance,
    std_dev,
    coefficient_variation,
    normalized_diameter,
    zscore
)
from .dynamics import (
    constriction_speed,
    dilation_speed,
    first_derivative,
    hippus_amplitude
)
from .area_ratio import pupil_iris_area_ratio

# Try to import Cython-accelerated versions
try:
    from ._metrics_cy import (
        variance as _variance_cy,
        std_dev as _std_dev_cy,
        coefficient_variation as _cv_cy,
        zscore as _zscore_cy
    )
    from ._dynamics_cy import (
        constriction_speed as _constriction_speed_cy,
        dilation_speed as _dilation_speed_cy,
        first_derivative as _first_derivative_cy,
        hippus_amplitude as _hippus_amplitude_cy
    )
    from ._area_ratio_cy import pupil_iris_area_ratio as _area_ratio_cy
    _HAS_CYTHON = True
except ImportError:
    _HAS_CYTHON = False

# If Cython is available, replace the Python functions with the accelerated ones
if _HAS_CYTHON:
    variance = _variance_cy
    std_dev = _std_dev_cy
    coefficient_variation = _cv_cy
    zscore = _zscore_cy
    constriction_speed = _constriction_speed_cy
    dilation_speed = _dilation_speed_cy
    first_derivative = _first_derivative_cy
    hippus_amplitude = _hippus_amplitude_cy
    pupil_iris_area_ratio = _area_ratio_cy

__all__ = [
    'extract_pupil_diameter',
    'extract_iris_radius',
    'variance',
    'std_dev',
    'coefficient_variation',
    'normalized_diameter',
    'zscore',
    'constriction_speed',
    'dilation_speed',
    'first_derivative',
    'hippus_amplitude',
    'pupil_iris_area_ratio',
]