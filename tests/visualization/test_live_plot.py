"""
Tests for visualization.live_plot module.
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Utilise backend non interactif pour les tests
from eyetrace.visualization.live_plot import LivePlot

def test_live_plot_initialization():
    """Test LivePlot initialization."""
    plot = LivePlot(max_points=100, title='Test')
    assert plot.max_points == 100
    assert plot.title == 'Test'

def test_live_plot_update():
    """Test updating plot with data."""
    plot = LivePlot(max_points=10)
    x = np.arange(5)
    y = np.random.randn(5)
    plot.update(x, y)
    # Vérifie que les données sont stockées
    assert len(plot.x_data) == 5
    assert len(plot.y_data) == 5

def test_live_plot_clear():
    """Test clearing plot."""
    plot = LivePlot()
    plot.update([1,2,3], [1,2,3])
    plot.clear()
    assert len(plot.x_data) == 0
    assert len(plot.y_data) == 0