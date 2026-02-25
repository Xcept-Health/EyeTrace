"""
Tests for signal_analysis.trend module.
"""

import pytest
import numpy as np
from eyetrace.signal_analysis.trend import trend_slope

def test_trend_slope():
    """Test linear trend slope."""
    t = np.arange(100)
    y = 2 * t + 5 + 0.1 * np.random.randn(100)
    slope = trend_slope(y, t)
    assert np.isclose(slope, 2.0, rtol=0.1)