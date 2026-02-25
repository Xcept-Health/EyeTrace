.. _io.hdf5_exporter:

HDF5 Exporter
=============

.. automodule:: eyetrace.io.hdf5_exporter
   :members:
   :undoc-members:
   :show-inheritance:

This module requires `h5py`. It enables efficient storage of large datasets
and hierarchical metadata.

- :class:`HDF5Exporter` : a context manager for creating HDF5 files,
  with methods to write datasets and attributes.

If `h5py` is not installed, the class will raise an `ImportError` upon
instantiation.