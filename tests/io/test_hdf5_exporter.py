"""
Tests for io.hdf5_exporter module.
"""

import pytest
import numpy as np
import h5py
import tempfile
import os
from eyetrace.io.hdf5_exporter import export_to_hdf5, import_from_hdf5

@pytest.mark.skipif(not h5py, reason="h5py not installed")
def test_export_import_hdf5():
    """Test export and import to HDF5."""
    data = {
        'timestamp': np.array([0.0, 0.1, 0.2]),
        'diameter': np.array([4.5, 4.6, 4.7]),
        'metadata': {'subject': '001', 'session': 1}
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, 'test.h5')
        export_to_hdf5(data, filepath)
        assert os.path.exists(filepath)
        imported = import_from_hdf5(filepath)
        np.testing.assert_allclose(imported['timestamp'], data['timestamp'])
        np.testing.assert_allclose(imported['diameter'], data['diameter'])
        assert imported['metadata']['subject'] == '001'
        assert imported['metadata']['session'] == 1