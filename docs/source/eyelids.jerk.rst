.. _eyelids-jerk:

EAR Jerk
========

.. module:: eyetrace.eyelids.jerk
   :synopsis: Rate of change of eyelid velocity.

Jerk is the derivative of acceleration, i.e., the second derivative of EAR. It measures how abruptly the eyelid movement changes, which can be useful for detecting micro‑saccades or spasms.

.. rubric:: Functions

.. automodule:: eyetrace.eyelids.jerk
   :members:
   :undoc-members:
   :show-inheritance:

.. rubric:: Example

.. code-block:: python

   import numpy as np
   from eyetrace.eyelids.jerk import ear_jerk

   # Simulated EAR signal (smooth)
   t = np.linspace(0, 2, 100)
   ear = 0.3 + 0.1 * np.sin(2 * np.pi * 2 * t)  # oscillating

   jerk = ear_jerk(ear, t)
   print("Jerk shape:", jerk.shape)