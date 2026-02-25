.. _eyelids-microsleep:

Microsleep Detection
====================

.. module:: eyetrace.eyelids.microsleep
   :synopsis: Detect episodes of prolonged eye closure.

Microsleeps are brief periods of involuntary sleep lasting from a fraction of a second to several seconds. They are characterized by prolonged eye closure (EAR below threshold for a duration exceeding a set threshold).

.. rubric:: Functions

.. automodule:: eyetrace.eyelids.microsleep
   :members:
   :undoc-members:
   :show-inheritance:

.. rubric:: Example

.. code-block:: python

   import numpy as np
   from eyetrace.eyelids.microsleep import microsleep_indicator

   # Simulate EAR signal with a 3-second microsleep at 30 fps
   ear = np.ones(300) * 0.3
   ear[100:190] = 0.1   # 90 frames = 3 seconds

   microsleep = microsleep_indicator(ear, frame_rate=30.0,
                                     ear_threshold=0.2,
                                     duration_threshold=2.0)
   print("Microsleep detected (True where duration >=2s):", np.any(microsleep))
   # The indicator is a boolean array marking frames within the microsleep.