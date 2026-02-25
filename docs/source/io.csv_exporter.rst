.. _io.csv_exporter:

CSV Exporter
============

.. automodule:: eyetrace.io.csv_exporter
   :members:
   :undoc-members:
   :show-inheritance:

The CSV exporter provides a convenient way to save time‑series data to
comma‑separated values files.

- :class:`CSVExporter` : a context manager that writes rows of data,
  automatically handling headers.
- :func:`save_metrics_to_csv` : a helper function to save aligned arrays
  (timestamps and multiple metrics) to a CSV file.io.hdf5_exporter