import pytest
import numpy as np
from eyetrace.pupil import extract_pupil_diameter, extract_iris_radius

def test_extract_pupil_diameter(sample_iris_landmarks):
    """Test l'extraction du diamètre pupillaire."""
    diam = extract_pupil_diameter(sample_iris_landmarks, 100, 100)
    # Avec des points sur un cercle parfait, diamètre ≈ 2 * rayon moyen
    assert np.isclose(diam, 20.0, rtol=0.1)
    
    # Test avec conversion en mm
    diam_mm = extract_pupil_diameter(sample_iris_landmarks, 100, 100, px_to_mm=0.05)
    assert np.isclose(diam_mm, 1.0, rtol=0.1)

def test_extract_iris_radius(sample_iris_landmarks):
    """Test l'extraction du rayon de l'iris."""
    radius = extract_iris_radius(sample_iris_landmarks, 100, 100)
    assert np.isclose(radius, 10.0, rtol=0.1)

def test_invalid_shape():
    """Test la gestion des formes incorrectes."""
    invalid = np.zeros((3, 2))
    with pytest.raises(ValueError):
        extract_pupil_diameter(invalid, 100, 100)