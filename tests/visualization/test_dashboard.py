"""
Tests for visualization.dashboard module.
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')
from eyetrace.visualization.dashboard import Dashboard

def test_dashboard_initialization():
    """Test Dashboard initialization."""
    dash = Dashboard(n_plots=3)
    assert dash.n_plots == 3

def test_dashboard_update():
    """Test updating dashboard with data."""
    dash = Dashboard(n_plots=2)
    data = [
        (np.arange(10), np.random.randn(10)),  # plot 1
        (np.arange(10), np.random.randn(10))   # plot 2
    ]
    # Doit s'exécuter sans erreur
    dash.update(data)
    # Pas d'assertion particulière, juste vérifier que ça ne plante pas
    assert True