import pytest
import numpy as np
from eyetrace.pupil import first_derivative, constriction_speed, dilation_speed

def test_first_derivative(sample_times):
    t = sample_times
    # Signal linéaire: y = 2t, dérivée = 2
    y = 2 * t
    deriv = first_derivative(y, t, smooth=False)
    np.testing.assert_allclose(deriv[1:-1], 2.0, rtol=0.1)

def test_constriction_speed():
    t = np.linspace(0, 5, 100)
    # Constriction de 5 à 3 entre t=1 et t=2
    d = np.ones_like(t) * 5
    mask = (t >= 1) & (t < 2)
    d[mask] = 5 - 2*(t[mask]-1)
    d[t >= 2] = 3
    
    max_speed, avg_speed = constriction_speed(d, t, threshold=0.5)
    assert max_speed < 0  # doit être négatif
    assert np.isclose(max_speed, -2.0, rtol=0.2)

def test_dilation_speed():
    t = np.linspace(0, 5, 100)
    # Dilatation de 3 à 5 entre t=1 et t=2
    d = np.ones_like(t) * 3
    mask = (t >= 1) & (t < 2)
    d[mask] = 3 + 2*(t[mask]-1)
    d[t >= 2] = 5
    
    max_speed, avg_speed = dilation_speed(d, t, threshold=0.5)
    assert max_speed > 0
    assert np.isclose(max_speed, 2.0, rtol=0.2)