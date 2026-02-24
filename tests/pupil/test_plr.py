import pytest
import numpy as np
from eyetrace.pupil.plr import plr_analysis, detect_constriction_onset

def test_detect_constriction_onset(sample_times, sample_diameters):
    stimulus = 3.0
    onset_idx, latency = detect_constriction_onset(
        sample_diameters, sample_times, stimulus
    )
    assert onset_idx is not None
    assert 0.1 < latency < 0.5  # latence typique

def test_plr_analysis(sample_times, sample_diameters):
    stimulus = 3.0
    results = plr_analysis(sample_diameters, sample_times, stimulus)
    
    assert 'baseline_mean' in results
    assert 'min_diameter' in results
    assert 'amplitude' in results
    assert results['amplitude'] > 0  # doit y avoir une constriction
    assert 'latency' in results
    assert results['latency'] > 0