Pupil Core
==========

.. automodule:: eyetrace.pupil.core
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

This module provides low-level functions to extract pupil diameter and iris radius from MediaPipe iris landmarks. These functions are the foundation for all higher-level pupil metrics.

Functions
---------

.. autofunction:: extract_pupil_diameter
.. autofunction:: extract_iris_radius

Details
-------

- The iris landmarks are expected to be in the format returned by MediaPipe Face Mesh with `refine_landmarks=True` (5 points per iris).
- If a conversion factor `px_to_mm` is provided, the result is converted to millimeters; otherwise, pixels are returned.
- Both functions raise `ValueError` if the input landmark array has an incorrect shape.