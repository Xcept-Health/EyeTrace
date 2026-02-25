.. _gaze.eccentricity:

Gaze Eccentricity
=================

.. automodule:: eyetrace.gaze.eccentricity
   :members:
   :undoc-members:
   :show-inheritance:

.. rubric:: Functions

- **pupil_eccentricity** : Computes the angular offset of the pupil relative to the optical axis.

.. rubric:: Notes

Pupil eccentricity is the angle between the line of sight (gaze direction) and the optical axis. It can be estimated from the displacement of the pupil center relative to the iris center. This is useful for correcting gaze estimates based on pupil position.

.. rubric:: Example

.. code-block:: python

    import numpy as np
    from eyetrace.gaze.eccentricity import pupil_eccentricity

    pupil_center = np.array([100, 100])
    iris_center = np.array([102, 100])
    gaze_vector = np.array([0, 0, 1])
    ecc = pupil_eccentricity(pupil_center, iris_center, gaze_vector)
    print(ecc)