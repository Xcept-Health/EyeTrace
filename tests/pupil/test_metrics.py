
import pytest
import numpy as np
from eyetrace.pupil import variance, std_dev, coefficient_variation

def test_variance():
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    # variance unbiased = 2.5
    assert np.isclose(variance(data), 2.5)
    # avec un seul élément
    assert variance([42.0]) == 0.0

def test_std_dev():
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert np.isclose(std_dev(data), np.sqrt(2.5))

def test_coefficient_variation():
    data = [10, 20, 30, 40, 50]
    # mean=30, std≈15.81, CV≈52.7%
    assert np.isclose(coefficient_variation(data), 52.7047, rtol=1e-3)
    # mean = 0
    assert coefficient_variation([0,0,0]) == 0.0