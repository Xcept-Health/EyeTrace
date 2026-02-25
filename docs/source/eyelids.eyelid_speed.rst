.. _eyelids-eyelid-speed:

Eyelid Speed
============

.. module:: eyetrace.eyelids.eyelid_speed
   :synopsis: Compute eyelid closing and opening speeds.

This module provides functions to compute the speed at which the eyelid closes (negative velocity) and opens (positive velocity) during a blink. The speed is derived from the EAR time series.

.. rubric:: Functions

.. automodule:: eyetrace.eyelids.eyelid_speed
   :members:
   :undoc-members:
   :show-inheritance:

.. rubric:: Example

.. code-block:: python

   import numpy as np
   from eyetrace.eyelids.eyelid_speed import eyelid_closing_speed, eyelid_opening_speed

   # Simulate EAR signal with a blink (30 fps)
   t = np.arange(0, 2, 1/30.0)
   ear = np.ones_like(t) * 0.3
   # linear closing from frame 20 to 30
   ear[20:30] = np.linspace(0.3, 0.1, 10)
   ear[30:40] = 0.1
   # linear opening from frame 40 to 50
   ear[40:50] = np.linspace(0.1, 0.3, 10)

   closing = eyelid_closing_speed(ear, t)
   opening = eyelid_opening_speed(ear, t)
   print("Closing speeds (max, avg):", closing)
   print("Opening speeds (max, avg):", opening)