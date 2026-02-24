"""
Shared fixtures for tests.
"""

import pytest
import numpy as np

@pytest.fixture
def sample_data():
    """Example data for tests."""
    return np.random.randn(100)