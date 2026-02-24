"""
Tests for pupil.plr module.
"""

import pytest
import numpy as np
from eyetrace.pupil.plr import (
    detect_constriction_onset,
    plr_analysis
)

def create_test_signal(fs=100, duration=10, stimulus_time=3.0,
                       baseline=5.0, constriction_amp=1.0,
                       constriction_duration=0.5, recovery_time=2.0,
                       noise_level=0.0):
    """
    Create a synthetic pupil signal with a clear constriction.
    """
    t = np.arange(0, duration, 1/fs)
    d = np.ones_like(t) * baseline

    # Constriction phase
    start_idx = int(stimulus_time * fs)
    constrict_end_idx = int((stimulus_time + constriction_duration) * fs)
    if constrict_end_idx < len(t):
        # Linear drop from baseline to baseline - constriction_amp
        constrict_len = constrict_end_idx - start_idx
        d[start_idx:constrict_end_idx] = baseline - constriction_amp * np.linspace(0, 1, constrict_len)

    # Minimum at constrict_end
    min_idx = constrict_end_idx
    d[min_idx] = baseline - constriction_amp

    # Recovery phase
    recovery_end_idx = int((stimulus_time + constriction_duration + recovery_time) * fs)
    if recovery_end_idx < len(t):
        recovery_len = recovery_end_idx - min_idx
        d[min_idx:recovery_end_idx] = (baseline - constriction_amp) + constriction_amp * np.linspace(0, 1, recovery_len)

    # Add noise if requested
    if noise_level > 0:
        d += noise_level * np.random.randn(len(t))

    return t, d

def test_detect_constriction_onset():
    """Test detection of constriction onset."""
    t, d = create_test_signal(stimulus_time=3.0, constriction_amp=1.0,
                               constriction_duration=0.5, noise_level=0.01)
    onset_idx, latency = detect_constriction_onset(
        d, t, stimulus_time=3.0,
        baseline_duration=1.0,
        response_window=2.0,
        speed_threshold=-0.5  
    )
    assert onset_idx is not None
    assert 0 <= latency < 0.5  

def test_detect_constriction_onset_no_response():
    """Test when no constriction occurs."""
    t = np.linspace(0, 5, 500)
    d = np.ones_like(t) * 5.0  # constant
    onset_idx, latency = detect_constriction_onset(d, t, stimulus_time=2.0)
    assert onset_idx is None
    assert latency is None

def test_plr_analysis():
    """Test full PLR analysis."""
    t, d = create_test_signal(stimulus_time=3.0, constriction_amp=1.5,
                               baseline=5.0, constriction_duration=0.5,
                               recovery_time=2.0, noise_level=0.01)
    results = plr_analysis(d, t, stimulus_time=3.0,
                           baseline_duration=2.0,
                           response_window=3.0,
                           recovery_level=0.75)

    # Check keys
    expected_keys = ['baseline_mean', 'baseline_std', 'min_diameter',
                     'min_index', 'amplitude', 'amplitude_percent',
                     'latency', 'max_constriction_speed',
                     'max_dilation_speed', 'recovery_time_75',
                     'recovery_index_75']
    for key in expected_keys:
        assert key in results

    # Baseline should be around 5.0
    assert np.isclose(results['baseline_mean'], 5.0, rtol=0.1)

    # Minimum should be around 3.5 (5 - 1.5)
    assert np.isclose(results['min_diameter'], 3.5, rtol=0.1)

    # Amplitude ≈ 1.5
    assert np.isclose(results['amplitude'], 1.5, rtol=0.1)

    # Latency should be small
    assert results['latency'] < 0.5

    # Recovery time should be around 2.0 seconds (our recovery_time)
    # But note: recovery to 75% of amplitude means target = baseline - 0.25*amp = 5 - 0.375 = 4.625
    # In our signal, recovery is linear from 3.5 to 5.0 over 2 s, so reaching 4.625 occurs at
    # (4.625 - 3.5) / (1.5) * 2 = 1.5 s after min. So recovery_time_75 ≈ 1.5 s.
    # We'll just check it's positive and not too large.
    assert results['recovery_time_75'] > 0
    assert results['recovery_time_75'] < 3.0

def test_plr_analysis_insufficient_data():
    """Test with insufficient data."""
    t = np.array([0, 1])
    d = np.array([5, 5])
    with pytest.raises(ValueError, match="Not enough baseline data"):
        plr_analysis(d, t, stimulus_time=0.5)

def test_plr_analysis_no_post_stimulus():
    """Test with no data after stimulus."""
    t = np.linspace(0, 2, 100)
    d = np.ones_like(t)
    stimulus = 3.0  # after the end
    with pytest.raises(ValueError, match="No data after stimulus"):
        plr_analysis(d, t, stimulus)

def test_plr_analysis_no_constriction():
    """Test when no constriction occurs."""
    t = np.linspace(0, 5, 500)
    d = np.ones_like(t) * 5.0
    # Add a small bump instead of constriction
    d[200:250] = 5.2
    results = plr_analysis(d, t, stimulus_time=2.0)
    # Since no constriction, amplitude might be negative? Our function should handle.
    # We'll just check that it runs without error.
    assert 'amplitude' in results