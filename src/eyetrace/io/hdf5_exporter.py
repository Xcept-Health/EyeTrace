"""
Export metrics to HDF5 files with flexible dataset handling.
"""

import numpy as np
from typing import Dict, Optional, Any, Union

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
        File mode: 'w' (write, overwrite file), 'a' (append), 'r+' (read/write).
    **kwargs
        Additional arguments passed to h5py.File (e.g., compression level).
    """
    def __init__(self, filename: str, mode: str = 'w', **kwargs):
        self.filename = filename
        self.mode = mode
        self.kwargs = kwargs
        self.file = None
        self._open_file()

    def _open_file(self):
        """Lazy import of h5py and open the file."""
        try:
            import h5py
        except ImportError as e:
            raise ImportError("h5py is not installed. Please install it with: pip install h5py") from e
        self.file = h5py.File(self.filename, self.mode, **self.kwargs)

    def write_array(self, name: str, data: np.ndarray,
                    compression: Optional[str] = 'gzip',
                    overwrite: bool = False,
                    **kwargs):
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
        overwrite : bool, default False
            If True and dataset exists, it will be deleted and recreated.
            If False and dataset exists, an error is raised.
        **kwargs
            Additional arguments passed to h5py.File.create_dataset.
        """
        if name in self.file:
            if overwrite:
                del self.file[name]
            else:
                raise ValueError(f"Dataset '{name}' already exists. Use overwrite=True to replace.")
        self.file.create_dataset(name, data=data, compression=compression, **kwargs)

    def write_attributes(self, name: str, attrs: Dict[str, Any]):
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

    def create_group(self, name: str, overwrite: bool = False):
        """
        Create a group.

        Parameters
        ----------
        name : str
            Path of the group.
        overwrite : bool, default False
            If True and group exists, it will be deleted and recreated.
        """
        if name in self.file:
            if overwrite:
                del self.file[name]
            else:
                raise ValueError(f"Group '{name}' already exists. Use overwrite=True to replace.")
        return self.file.create_group(name)

    def close(self):
        """Close the HDF5 file."""
        if self.file is not None:
            self.file.close()
            self.file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()