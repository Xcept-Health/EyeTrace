"""
Tests for io.csv_exporter module.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from eyetrace.io.csv_exporter import export_to_csv, import_from_csv

def test_export_import_csv():
    """Test export and import of data to CSV."""
    data = {
        'timestamp': np.array([0.0, 0.1, 0.2]),
        'diameter': np.array([4.5, 4.6, 4.7]),
        'blink': np.array([0, 1, 0])
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, 'test.csv')
        export_to_csv(data, filepath)
        # Vérifier que le fichier existe
        assert os.path.exists(filepath)
        # Importer et vérifier
        imported = import_from_csv(filepath)
        np.testing.assert_allclose(imported['timestamp'], data['timestamp'])
        np.testing.assert_allclose(imported['diameter'], data['diameter'])
        np.testing.assert_allclose(imported['blink'], data['blink'])

def test_export_empty():
    """Test export with empty data."""
    data = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, 'empty.csv')
        with pytest.raises(ValueError):
            export_to_csv(data, filepath)