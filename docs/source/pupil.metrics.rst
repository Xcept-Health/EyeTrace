Pupil Metrics
=============

.. automodule:: eyetrace.pupil.metrics
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

This module contains basic statistical metrics for pupil diameter time series. All functions are implemented in pure Python with an optional Cython-accelerated version for better performance.

Functions
---------

.. autofunction:: variance
.. autofunction:: std_dev
.. autofunction:: coefficient_variation
.. autofunction:: normalized_diameter
.. autofunction:: zscore

Mathematical Definitions
------------------------

- **Variance**: :math:`\sigma^2 = \frac{1}{N-1}\sum_{i=1}^N (d_i - \bar{d})^2`
- **Standard deviation**: :math:`\sigma = \sqrt{\sigma^2}`
- **Coefficient of variation**: :math:`CV = \frac{\sigma}{\bar{d}} \times 100\%`
- **Normalized diameter**: :math:`d_{\text{norm}} = \frac{d}{d_{\text{ref}}}`
- **Z‑score**: :math:`z = \frac{d - \bar{d}}{\sigma}`

Notes
-----

- All functions accept any array-like input and return a float or numpy array as appropriate.
- For sequences with fewer than 2 elements, variance, standard deviation and CV return 0.0.
- `normalized_diameter` raises a `ValueError` if the reference diameter is zero.