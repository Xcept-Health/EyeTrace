Installation
============

Prerequisites
-------------

- Python 3.8 or higher
- pip (Python package installer)

From PyPI (soon)
----------------

.. code-block:: bash

   pip install eyetrace

From source (latest development version)
-----------------------------------------

Clone the repository:

.. code-block:: bash

   git clone https://github.com/Xcept-Health/EyeTrace.git
   cd EyeTrace

Install in development mode (editable):

.. code-block:: bash

   pip install -e .

This will automatically install all required dependencies:

- numpy
- scipy
- opencv-python
- mediapipe
- pandas
- matplotlib
- cython (for building extensions)

Optional dependencies
---------------------

For GPU acceleration with MediaPipe, follow the `official instructions <https://google.github.io/mediapipe/getting_started/python.html>`_.

For HDF5 export, install ``h5py``:

.. code-block:: bash

   pip install h5py

For running tests and benchmarks:

.. code-block:: bash

   pip install pytest pytest-benchmark pytest-cov

For building documentation:

.. code-block:: bash

   pip install sphinx sphinx_rtd_theme numpydoc