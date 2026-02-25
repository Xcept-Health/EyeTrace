.. _gaze.saccades:

Gaze Saccades
=============

.. automodule:: eyetrace.gaze.saccades
   :members:
   :undoc-members:
   :show-inheritance:

.. rubric:: Functions

- **detect_saccades** : Identifies saccadic events based on velocity threshold.
- **saccade_velocity** : Computes instantaneous angular velocity of gaze.
- **saccade_acceleration** : Computes angular acceleration from velocity.
- **saccade_fixation_ratio** : Calculates ratio of time spent in saccades vs fixations.

.. rubric:: Notes

Saccades are rapid eye movements that shift the fovea to a new target. Detection typically uses a velocity threshold (e.g., >30 deg/s). The module assumes input gaze positions are 3D vectors (e.g., from eye tracking) and timestamps are provided.

.. rubric:: Example

.. code-block:: python

    import numpy as np
    from eyetrace.gaze.saccades import detect_saccades, saccade_velocity

    # Simulated gaze data (3D vectors)
    t = np.linspace(0, 10, 1000)
    gaze = np.zeros((1000, 3))
    # Add a saccade
    gaze[500:510, 0] = np.linspace(0, 10, 10)
    saccades = detect_saccades(gaze, t, velocity_threshold=30.0)
    print(saccades)