"""
Tests for pupil.area_ratio module.
"""

import pytest
import numpy as np
from eyetrace.pupil import pupil_iris_area_ratio

def test_pupil_iris_area_ratio():
    """Test area ratio calculation."""
    ratio = pupil_iris_area_ratio(4.0, 8.0)
    assert np.isclose(ratio, 0.25)  # (4/8)^2 = 0.25

    ratio2 = pupil_iris_area_ratio(2.0, 4.0)
    assert np.isclose(ratio2, 0.25)

    # Zero iris diameter
    with pytest.raises(ValueError, match="iris_diameter cannot be zero"):
        pupil_iris_area_ratio(4.0, 0.0)

    # Zero pupil diameter is fine
    ratio_zero = pupil_iris_area_ratio(0.0, 5.0)
    assert ratio_zero == 0.0