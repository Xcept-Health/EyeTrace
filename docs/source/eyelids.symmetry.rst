.. _eyelids-symmetry:

Eyelid Symmetry
===============

.. module:: eyetrace.eyelids.symmetry
   :synopsis: Measure correlation between left and right eye EAR.

This module computes the symmetry between left and right eyelid movements. A high correlation indicates synchronous blinking, while low correlation may indicate neurological issues.

.. rubric:: Functions

.. automodule:: eyetrace.eyelids.symmetry
   :members:
   :undoc-members:
   :show-inheritance:

.. rubric:: Example

.. code-block:: python

   import numpy as np
   from eyetrace.eyelids.symmetry import eyelid_symmetry

   # Simulated left and right EAR signals
   left = np.array([0.3, 0.3, 0.2, 0.2, 0.3, 0.3])
   right = np.array([0.3, 0.3, 0.2, 0.2, 0.3, 0.3])

   corr = eyelid_symmetry(left, right)
   print(f"Symmetry (correlation): {corr:.3f}")