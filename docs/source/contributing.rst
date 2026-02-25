.. _contributing:

Contributing
============

We warmly welcome contributions to EyeTrace! Whether you are fixing bugs,
adding new features, improving documentation, or suggesting ideas, your help
is appreciated.

Please take a moment to review this guide before submitting any changes.

Code of Conduct
---------------

This project adheres to a `Code of Conduct <https://github.com/Xcept-Health/EyeTrace/blob/main/CODE_OF_CONDUCT.md>`_.
By participating, you are expected to uphold this code. Please report
unacceptable behavior to the project maintainers.

How to Contribute
-----------------

1. **Fork the repository** on GitHub.
2. **Clone your fork** locally:
   .. code-block:: bash

      git clone https://github.com/your-username/EyeTrace.git
      cd EyeTrace

3. **Create a branch** for your changes:
   .. code-block:: bash

      git checkout -b feature/your-feature-name

4. **Install the package in development mode** with extra dependencies:
   .. code-block:: bash

      pip install -e .[dev]

5. **Make your changes**, following the coding conventions (see below).
6. **Write tests** for any new functionality or bug fixes. Place tests in the
   appropriate subdirectory under ``tests/``.
7. **Run the tests** to ensure everything passes:
   .. code-block:: bash

      pytest tests/

8. **Update documentation** if necessary. If you add a new function or module,
   please update the corresponding ``.rst`` files in ``docs/source/``.
9. **Commit your changes** with a clear and descriptive commit message.
10. **Push to your fork** and open a pull request against the ``main`` branch
    of the original repository.

Coding Conventions
------------------

- **Python**: Follow `PEP 8 <https://peps.python.org/pep-0008/>`_. Use
  descriptive variable names.
- **Docstrings**: Use the NumPy/SciPy format for docstrings. Example:

  .. code-block:: python

     def example_function(param1, param2):
         """
         Short description.

         Parameters
         ----------
         param1 : type
             Description.
         param2 : type
             Description.

         Returns
         -------
         result : type
             Description.
         """
- **Type hints**: Add type annotations where possible, especially for public
  functions.
- **Cython**: For performance‑critical functions, provide a Cython version in a
  separate ``_*.pyx`` file and include a pure Python fallback.

Testing
-------

- All tests are located in the ``tests/`` directory, mirroring the structure of
  ``src/eyetrace/``.
- We use ``pytest`` for testing. Install it with ``pip install pytest``.
- Run the full test suite with ``pytest tests/``.
- To measure coverage, install ``pytest-cov`` and run:
  ``pytest tests/ --cov=eyetrace --cov-report=html``

Documentation
-------------

Documentation is built with Sphinx. To build it locally:

.. code-block:: bash

   cd docs
   pip install -e ..[docs]
   sphinx-build -b html source build

Then open ``docs/build/index.html`` in your browser.

If you add a new module, create a corresponding ``.rst`` file under
``docs/source/`` and include it in the appropriate toctree.

Pull Request Process
--------------------

- Ensure all tests pass and the documentation builds without errors.
- Update the ``Roadmap.md`` if your change affects the project plan.
- Your pull request will be reviewed by maintainers. Please be responsive to
  feedback.
- Once approved, it will be merged.

Thank you for contributing to EyeTrace!