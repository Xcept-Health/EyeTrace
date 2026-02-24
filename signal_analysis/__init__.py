"""
Advanced signal analysis for physiological time series.

This module provides functions for spectral analysis (FFT, power ratios),
complexity measures (sample entropy, Lempel-Ziv, Higuchi fractal dimension),
memory analysis (Hurst exponent), synchronization (mutual information),
signal quality (SNR), trend estimation, and a combined fatigue score (KSS).
"""

from .fourier import power_spectrum, band_power, lf_hf_ratio
from .entropy import sample_entropy, approximate_entropy
from .hurst import hurst_exponent
from .lempel_ziv import lempel_ziv_complexity
from .higuchi import higuchi_fractal_dimension
from .mutual_info import mutual_information
from .snr import signal_to_noise_ratio
from .trend import trend_slope
from .kss import karlinska_sleepiness_score

# Try to import Cython-accelerated versions for heavy functions
try:
    from ._entropy_cy import sample_entropy as _sample_entropy_cy
    _HAS_CYTHON_ENTROPY = True
except ImportError:
    _HAS_CYTHON_ENTROPY = False

try:
    from ._hurst_cy import hurst_exponent as _hurst_cy
    _HAS_CYTHON_HURST = True
except ImportError:
    _HAS_CYTHON_HURST = False

try:
    from ._lempel_ziv_cy import lempel_ziv_complexity as _lz_cy
    _HAS_CYTHON_LZ = True
except ImportError:
    _HAS_CYTHON_LZ = False

try:
    from ._higuchi_cy import higuchi_fractal_dimension as _higuchi_cy
    _HAS_CYTHON_HIGUCHI = True
except ImportError:
    _HAS_CYTHON_HIGUCHI = False

# Replace with Cython versions if available
if _HAS_CYTHON_ENTROPY:
    sample_entropy = _sample_entropy_cy
if _HAS_CYTHON_HURST:
    hurst_exponent = _hurst_cy
if _HAS_CYTHON_LZ:
    lempel_ziv_complexity = _lz_cy
if _HAS_CYTHON_HIGUCHI:
    higuchi_fractal_dimension = _higuchi_cy

__all__ = [
    'power_spectrum',
    'band_power',
    'lf_hf_ratio',
    'sample_entropy',
    'approximate_entropy',
    'hurst_exponent',
    'lempel_ziv_complexity',
    'higuchi_fractal_dimension',
    'mutual_information',
    'signal_to_noise_ratio',
    'trend_slope',
    'karlinska_sleepiness_score',
]