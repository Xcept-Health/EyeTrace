.. _eyelids-ear:

Eye Aspect Ratio (EAR)
======================

.. module:: eyetrace.eyelids.ear
   :synopsis: Functions for computing the Eye Aspect Ratio.

This module provides functions to compute the Eye Aspect Ratio (EAR) from eye landmarks. EAR is a scalar quantity that reflects the openness of the eye; it decreases when the eye closes.

**Formula**:

.. math::

   EAR = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2 \cdot \|p_1 - p_4\|}

where :math:`p_1, p_2, \dots, p_6` are the six eye landmarks in the order:
- p1: outer corner
- p2: top outer point
- p3: top inner point
- p4: inner corner
- p5: bottom inner point
- p6: bottom outer point

.. rubric:: Functions

.. automodule:: eyetrace.eyelids.ear
   :members:
   :undoc-members:
   :show-inheritance:

.. rubric:: Example

.. code-block:: python

   import numpy as np
   from eyetrace.eyelids.ear import eye_aspect_ratio, both_eyes_ear

   # Simulated landmarks for left eye (open)
   left_eye = np.array([
       [30,30], [32,25], [38,25], [40,30], [38,35], [32,35]
   ], dtype=np.float64)

   # Simulated landmarks for right eye (open)
   right_eye = np.array([
       [60,30], [62,25], [68,25], [70,30], [68,35], [62,35]
   ], dtype=np.float64)

   ear_left = eye_aspect_ratio(left_eye)
   ear_right = eye_aspect_ratio(right_eye)
   ear_avg = both_eyes_ear(left_eye, right_eye)

   print(f"Left EAR: {ear_left:.3f}")
   print(f"Right EAR: {ear_right:.3f}")
   print(f"Average EAR: {ear_avg:.3f}")