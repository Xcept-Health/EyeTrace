.. _gaze.vergence:

Gaze Vergence
=============

.. automodule:: eyetrace.gaze.vergence
   :members:
   :undoc-members:
   :show-inheritance:

.. rubric:: Functions

- **vergence_angle** : Computes the angle between left and right gaze vectors.
- **vergence_speed** : Computes the rate of change of vergence angle.

.. rubric:: Notes

Vergence is the simultaneous movement of both eyes in opposite directions to maintain single binocular vision. The module expects 3D gaze vectors for left and right eyes, and timestamps.

.. rubric:: Example

.. code-block:: python

    import numpy as np
    from eyetrace.gaze.vergence import vergence_angle, vergence_speed

    # Simulated left and right gaze vectors
    left = np.array([[0,0,1], [0.1,0,0.99]])
    right = np.array([[0,0,1], [-0.1,0,0.99]])
    t = np.array([0, 0.1])
    angle = vergence_angle(left, right)
    speed = vergence_speed(left, right, t)