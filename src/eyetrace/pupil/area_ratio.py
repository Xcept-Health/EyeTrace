"""
Pupil-to-iris area ratio.
"""

import numpy as np

def pupil_iris_area_ratio(
    pupil_diameter: float,
    iris_diameter: float
) -> float:
    """
    Compute the ratio of pupil area to iris area.

    Areas are approximated as circles: area = π * (d/2)^2.
    The ratio simplifies to (pupil_diameter / iris_diameter)^2.

    Parameters
    ----------
    pupil_diameter : float
        Diameter of the pupil (in any unit).
    iris_diameter : float
        Diameter of the iris (same unit).

    Returns
    -------
    float
        Area ratio (pupil/iris).
    """
    if iris_diameter == 0:
        raise ValueError("iris_diameter cannot be zero")
    return (pupil_diameter / iris_diameter) ** 2