.. _eyelids-perclos:

PERCLOS
=======

.. module:: eyetrace.eyelids.perclos
   :synopsis: Percentage of eyelid closure over time.

PERCLOS (Percentage of Eyelid Closure) is a standard metric for drowsiness detection. It measures the proportion of time that the eyes are closed (EAR below a threshold) over a sliding window, typically 60 seconds.

.. rubric:: Functions

.. automodule:: eyetrace.eyelids.perclos
   :members:
   :undoc-members:
   :show-inheritance:

.. rubric:: Example

.. code-block:: python

   import numpy as np
   from eyetrace.eyelids.perclos import perclos

   # Simulated EAR signal: 300 frames at 30 fps = 10 seconds
   ear = np.ones(300) * 0.3
   ear[100:150] = 0.1   # closed for 50 frames (1.67 s)

   p = perclos(ear, threshold=0.2, window_seconds=10.0, frame_rate=30.0)
   print(f"PERCLOS over 10s window: {p:.2f}%")