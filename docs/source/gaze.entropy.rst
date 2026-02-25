.. _gaze.entropy:

Gaze Entropy
============

.. automodule:: eyetrace.gaze.entropy
   :members:
   :undoc-members:
   :show-inheritance:

.. rubric:: Functions

- **gaze_entropy** : Computes Shannon entropy of gaze positions discretized into a 2D histogram.

.. rubric:: Notes

Gaze entropy quantifies the randomness or dispersion of gaze points. High entropy indicates exploratory behavior, while low entropy suggests focused attention. The function accepts a 2D array of gaze positions (x, y) and optional bin parameters.

.. rubric:: Example

.. code-block:: python

    import numpy as np
    from eyetrace.gaze.entropy import gaze_entropy

    # Simulated gaze points
    points = np.random.rand(1000, 2)
    entropy = gaze_entropy(points, bins=20)
    print(entropy)