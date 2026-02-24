"""
Export metrics to CSV files.
"""

import csv
import numpy as np
from typing import Dict, List, Any, Optional

class CSVExporter:
    """
    Write metrics to a CSV file, one row per timestamp.

    Parameters
    ----------
    filename : str
        Path to the output CSV file.
    fieldnames : list of str, optional
        Column names. If not given, they will be inferred from the first row.
    """
    def __init__(self, filename: str, fieldnames: Optional[List[str]] = None):
        self.filename = filename
        self.file = open(filename, 'w', newline='')
        self.writer = None
        self.fieldnames = fieldnames
        self._header_written = False

    def write_row(self, data: Dict[str, Any]):
        """
        Write a single row of data.

        Parameters
        ----------
        data : dict
            Dictionary with keys as column names and values as numbers/strings.
        """
        if not self._header_written:
            if self.fieldnames is None:
                self.fieldnames = list(data.keys())
            self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
            self.writer.writeheader()
            self._header_written = True
        # Ensure all keys exist, fill missing with empty string
        row = {k: data.get(k, '') for k in self.fieldnames}
        self.writer.writerow(row)

    def write_rows(self, rows: List[Dict[str, Any]]):
        """Write multiple rows."""
        for row in rows:
            self.write_row(row)

    def close(self):
        """Close the CSV file."""
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def save_metrics_to_csv(filename: str, timestamps: np.ndarray,
                        metrics_dict: Dict[str, np.ndarray]):
    """
    Convenience function to save a set of metrics aligned in time.

    Parameters
    ----------
    filename : str
        Output CSV file.
    timestamps : np.ndarray
        1D array of timestamps.
    metrics_dict : dict
        Dictionary mapping metric names to 1D arrays of same length.
    """
    with CSVExporter(filename) as exporter:
        for i, t in enumerate(timestamps):
            row = {'timestamp': t}
            for name, arr in metrics_dict.items():
                row[name] = arr[i]
            exporter.write_row(row)