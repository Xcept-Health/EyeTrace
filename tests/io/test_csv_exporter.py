"""
Tests for io.csv_exporter module.
"""

import pytest
import numpy as np
import tempfile
import os
import csv
from eyetrace.io.csv_exporter import CSVExporter, save_metrics_to_csv

def test_csv_exporter():
    """Test CSVExporter class."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, 'test.csv')
        exporter = CSVExporter(filepath, fieldnames=['timestamp', 'diameter', 'blink'])
        exporter.write_row({'timestamp': 0.0, 'diameter': 4.5, 'blink': 0})
        exporter.write_row({'timestamp': 0.1, 'diameter': 4.6, 'blink': 1})
        exporter.close()

        # Check that the file exists
        assert os.path.exists(filepath)
        
        # Read and verify content
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 2
            assert rows[0]['timestamp'] == '0.0'
            assert rows[0]['diameter'] == '4.5'
            assert rows[0]['blink'] == '0'

def test_save_metrics_to_csv():
    """Test save_metrics_to_csv convenience function."""
    data = {
        'timestamp': np.array([0.0, 0.1, 0.2]),
        'diameter': np.array([4.5, 4.6, 4.7]),
        'blink': np.array([0, 1, 0])
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, 'test.csv')
        save_metrics_to_csv(filepath, data['timestamp'],
                            {'diameter': data['diameter'], 'blink': data['blink']})
        assert os.path.exists(filepath)
        
        # Read and verify content
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 3
            assert rows[1]['timestamp'] == '0.1'
            assert rows[1]['diameter'] == '4.6'
            assert rows[1]['blink'] == '1'
