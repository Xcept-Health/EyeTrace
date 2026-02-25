Pupil Dynamics
==============

.. automodule:: eyetrace.pupil.dynamics
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

This module provides functions for analyzing the dynamic behaviour of the pupil, including its rate of change, constriction and dilation speeds, and spontaneous oscillations (hippus).

Functions
---------

.. autofunction:: first_derivative
.. autofunction:: constriction_speed
.. autofunction:: dilation_speed
.. autofunction:: hippus_amplitude

Algorithm Details
-----------------

**First derivative** is computed using central differences with optional Savitzky–Golay smoothing to reduce noise. If timestamps are not provided, uniform sampling (dt = 1) is assumed.

**Constriction and dilation speeds** are determined by identifying the most negative (constriction) or most positive (dilation) velocity segments that exceed a given duration threshold. The function returns both the maximum speed and the average speed during the corresponding phase.

**Hippus amplitude** is estimated by bandpass filtering the signal in the typical hippus frequency range (0.5–4 Hz) and then calculating either the root‑mean‑square (RMS) of the filtered signal or the mean of its Hilbert envelope.

Parameters
----------

All functions accept `diameters` (array) and `times` (array) as primary inputs. Optional parameters control smoothing, thresholds, and analysis methods.

Returns
-------

- `first_derivative` returns an array of velocities (same length as input).
- `constriction_speed` and `dilation_speed` return a tuple `(max_speed, avg_speed)`.
- `hippus_amplitude` returns a single float representing the amplitude.