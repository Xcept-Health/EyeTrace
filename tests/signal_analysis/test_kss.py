"""
Tests for signal_analysis.kss module.
"""

import pytest
import numpy as np
from eyetrace.signal_analysis.kss import karlinska_sleepiness_score

def test_kss_prediction():
    """Test KSS prediction (placeholder)."""
    # Simulate a dictionary of physiological and behavioral features
    features = {
        'perclos': 15.0,
        'blink_frequency': 20.0,
        'pupil_variance': 0.5,
        'head_movement': 0.1
    }
    kss = karlinska_sleepiness_score(features)
    # Ensure the Karolinska Sleepiness Scale score is within valid range [1, 9]
    assert 1 <= kss <= 9

    # Test with an empty dictionary (should return a default baseline value)
    kss2 = karlinska_sleepiness_score({})
    assert 1 <= kss2 <= 9
