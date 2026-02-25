Pupil‑Iris Area Ratio
=====================

.. automodule:: eyetrace.pupil.area_ratio
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

A single function that computes the ratio of the pupil area to the iris area. Both are approximated as circles, so the ratio simplifies to the square of the diameter ratio.

.. math::
   \text{ratio} = \left(\frac{d_{\text{pupil}}}{d_{\text{iris}}}\right)^2

Function
--------

.. autofunction:: pupil_iris_area_ratio

Notes
-----

- The function raises `ValueError` if `iris_diameter` is zero.
- The result is dimensionless.