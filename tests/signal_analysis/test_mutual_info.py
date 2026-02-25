"""
Tests for signal_analysis.mutual_info module.
"""

import pytest
import numpy as np
from eyetrace.signal_analysis.mutual_info import mutual_information

def test_mutual_information():
    """Test mutual information between two signals."""
    x = np.random.randn(1000)
    # Strongly correlated signals
    y = x + 0.1 * np.random.randn(1000)  
    mi = mutual_information(x, y, bins=20)
    assert mi > 0.5

    # Independent signals
    z = np.random.randn(1000)
    mi2 = mutual_information(x, z, bins=20)
    # Mutual information should be significantly lower for independent signals
    assert mi2 < 0.2
