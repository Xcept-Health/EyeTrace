Pupillary Light Reflex (PLR)
=============================

.. automodule:: eyetrace.pupil.plr
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

This module provides a complete analysis of the pupillary light reflex. Given a stimulus time, it computes baseline characteristics, constriction amplitude, latency, maximum constriction and dilation speeds, and the time to recover 75% of the baseline diameter.

Functions
---------

.. autofunction:: detect_constriction_onset
.. autofunction:: plr_analysis

Algorithm Details
-----------------

**detect_constriction_onset** searches for the first point after the stimulus where the pupil velocity falls below a threshold (negative), indicating the start of constriction. The latency is the time difference between stimulus and that point.

**plr_analysis** performs the full analysis:

1. **Baseline**: Mean and standard deviation of pupil diameters over a configurable period before the stimulus.
2. **Minimum**: The smallest diameter after the stimulus (within a reasonable window).
3. **Amplitude**: Baseline minus minimum diameter (absolute and percent).
4. **Latency**: Time from stimulus to constriction onset (if detected).
5. **Maximum constriction speed**: Most negative velocity between onset and the minimum.
6. **Maximum dilation speed**: Most positive velocity after the minimum.
7. **Recovery time**: Time from the minimum to when the diameter returns to a given fraction of the baseline (default 75%).

All outputs are returned in a dictionary.

Example
-------

.. code-block:: python

   import numpy as np
   from eyetrace.pupil.plr import plr_analysis

   t = np.linspace(0, 10, 1000)
   d = 5.0 - 1.5 * np.exp(-(t-3)**2/0.5)  # synthetic constriction at t=3
   results = plr_analysis(d, t, stimulus_time=3.0)

   print(f"Latency: {results['latency']:.3f} s")
   print(f"Amplitude: {results['amplitude']:.2f} px ({results['amplitude_percent']:.1f}%)")