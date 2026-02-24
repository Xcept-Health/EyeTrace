"""
Export metrics to HDF5 files.
"""

import numpy as np
from typing import Dict, Optional

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False


class HDF5Exporter:
    """
    Write metrics to an HDF5 file.

    This is useful for large datasets and hierarchical organization.
    Requires h5py.

    Parameters
    ----------
    filename : str
        Path to the output HDF5 file.
    mode : str, default 'w'
        File mode: 'w' (write, overwrite), 'a' (append), etc.
    """
    def __init__(self, filename: str, mode: str = 'w'):
        if not H5PY_AVAILABLE:
            raise ImportError("h5py is not installed. Please install it with: pip install h5py")
        self.filename = filename
        self.file = h5py.File(filename, mode)

    def write_array(self, name: str, data: np.ndarray,
                    compression: Optional[str] = 'gzip', **kwargs):
        """
        Write a dataset to the file.

        Parameters
        ----------
        name : str
            Path/name of the dataset (e.g., 'pupil/diameter').
        data : np.ndarray
            Array to store.
        compression : str or None
            Compression filter (e.g., 'gzip'). None for no compression.
        **kwargs
            Additional arguments passed to h5py.File.create_dataset.
        """
        self.file.create_dataset(name, data=data, compression=compression, **kwargs)

    def write_attributes(self, name: str, attrs: Dict[str, any]):
        """
        Write attributes to a dataset or group.

        Parameters
        ----------
        name : str
            Path to the dataset or group.
        attrs : dict
            Dictionary of attributes.
        """
        obj = self.file[name]
        for key, value in attrs.items():
            obj.attrs[key] = value

    def close(self):
        """Close the HDF5 file."""
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()