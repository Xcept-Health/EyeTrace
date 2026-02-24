import pytest
from eyetrace.pupil import pupil_iris_area_ratio

def test_pupil_iris_area_ratio():
    ratio = pupil_iris_area_ratio(4.0, 8.0)
    assert np.isclose(ratio, 0.25)
    
    with pytest.raises(ValueError):
        pupil_iris_area_ratio(4.0, 0.0)