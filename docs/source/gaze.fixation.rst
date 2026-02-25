.. _gaze.fixation:

Gaze Fixation
=============

.. automodule:: eyetrace.gaze.fixation
   :members:
   :undoc-members:
   :show-inheritance:

.. rubric:: Functions

- **fixation_duration** : Returns list of fixation durations from a fixation mask.
- **fixation_dispersion** : Computes spatial dispersion (standard deviation) for each fixation.
- **gaze_centroid** : Calculates the mean gaze position over a set of points.

.. rubric:: Notes

Fixations are periods when the eye remains relatively still, allowing visual processing. The module expects a boolean mask indicating fixation frames (e.g., from a classifier) and timestamps.

.. rubric:: Example

.. code-block:: python

    import numpy as np
    from eyetrace.gaze.fixation import fixation_duration, fixation_dispersion

    # Simulated fixation mask and timestamps
    mask = np.array([1,1,1,0,0,1,1,1,1])
    t = np.arange(len(mask))
    durations = fixation_duration(mask, t)
    print(durations)