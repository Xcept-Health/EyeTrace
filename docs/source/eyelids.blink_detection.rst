.. _eyelids-blink-detection:

Blink Detection
===============

.. module:: eyetrace.eyelids.blink_detection
   :synopsis: Detect blinks and compute blink statistics.

This module provides functions to detect blinks from an EAR time series and compute statistics such as blink frequency, mean closure duration, and ratio of long blinks.

**Algorithm**: Blinks are identified as periods where EAR drops below a threshold for at least two consecutive frames. Adjacent blinks closer than a minimum interval can be merged.

.. rubric:: Functions

.. automodule:: eyetrace.eyelids.blink_detection
   :members:
   :undoc-members:
   :show-inheritance:

.. rubric:: Example

.. code-block:: python

   import numpy as np
   from eyetrace.eyelids.blink_detection import detect_blinks, blink_frequency, mean_closure_duration

   # Simulated EAR signal with two blinks
   ear = np.ones(100) * 0.3
   ear[20:25] = 0.1   # blink 1: frames 20-24
   ear[60:63] = 0.1   # blink 2: frames 60-62

   blinks = detect_blinks(ear, threshold=0.2)
   print("Blink intervals (start, end):", blinks)

   duration = 100 / 30.0  # assuming 30 fps
   freq = blink_frequency(blinks, duration)
   print(f"Blink frequency: {freq:.2f} blinks/min")

   mcd = mean_closure_duration(blinks, frame_rate=30.0)
   print(f"Mean closure duration: {mcd:.3f} s")