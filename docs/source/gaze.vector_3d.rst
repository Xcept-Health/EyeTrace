.. _gaze.vector_3d:

Gaze Vector 3D
==============

.. automodule:: eyetrace.gaze.vector_3d
   :members:
   :undoc-members:
   :show-inheritance:

.. rubric:: Functions

- **gaze_vector_3d** : Estimates 3D gaze direction from eye landmarks (placeholder).

.. rubric:: Notes

This function is a placeholder for a more sophisticated model that would combine eye landmarks and head pose to compute a 3D gaze vector. Currently returns a dummy vector pointing straight ahead. For real applications, consider integrating with geometric eye models.

.. rubric:: Example

.. code-block:: python

    import numpy as np
    from eyetrace.gaze.vector_3d import gaze_vector_3d

    # Dummy eye landmarks (6 points)
    eye = np.random.rand(6, 2)
    vec = gaze_vector_3d(eye)
    print(vec)