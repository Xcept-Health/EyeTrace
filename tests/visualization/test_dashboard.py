"""
Tests for visualization.dashboard module.
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')
from eyetrace.visualization.dashboard import Dashboard

# Mock video source
class MockVideoSource:
    def __iter__(self):
        # Yield a blank frame for simulation
        yield np.zeros((480, 640, 3), dtype=np.uint8)

def test_dashboard_initialization():
    """Test Dashboard initialization."""
    video = MockVideoSource()
    plot_specs = [{'title': 'Metric 1', 'ylabel': 'value'},
                  {'title': 'Metric 2', 'ylabel': 'value'}]
    dash = Dashboard(video_source=video, plot_specs=plot_specs, update_interval_ms=50)
    
    # Check if video source is correctly assigned and plot specs are registered
    assert hasattr(dash, 'video')
    assert len(dash.plot_specs) == 2

def test_dashboard_update():
    """Test updating dashboard with data."""
    video = MockVideoSource()
    plot_specs = [{'title': 'Metric 1', 'ylabel': 'value'},
                  {'title': 'Metric 2', 'ylabel': 'value'}]
    dash = Dashboard(video, plot_specs)
    
    # Simulate a dashboard update cycle
    dash.update_plots(1.0, [0.5, 0.3])
    
    # Ensure update runs without raising exceptions
    assert True
