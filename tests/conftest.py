"""Fixtures partagées pour les tests."""

import pytest
import numpy as np

@pytest.fixture
def sample_data():
    """Données d'exemple pour les tests."""
    return np.random.randn(100)
