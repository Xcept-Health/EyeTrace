"""
Tests for signal_analysis.trend module.
"""

import pytest
import numpy as np
from eyetrace.signal_analysis.trend import trend_slope

def test_trend_slope():
    """Test linear trend slope."""
    t = np.arange(100)
    # Define a linear trend with a slope of 2.0 and some noise
    y = 2 * t + 5 + 0.1 * np.random.randn(100)
    slope = trend_slope(y, t)
    # The estimated slope should be close to 2.0
    assert np.isclose(slope, 2.0, rtol=0.1)
