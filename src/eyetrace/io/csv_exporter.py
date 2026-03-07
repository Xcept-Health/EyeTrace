"""
Export metrics to CSV files with metadata and buffered writing.
"""

import csv
import numpy as np
from typing import Dict, List, Any, Optional


class CSVExporter:
    """
    Write metrics to a CSV file, one row per timestamp.

    Supports writing metadata as comments at the top of the file,
    and optional buffering for better performance when writing many rows.

    Parameters
    ----------
    filename : str
        Path to the output CSV file.
    fieldnames : list of str, optional
        Column names. If not given, they will be inferred from the first row.
    metadata : dict, optional
        Global metadata to write as comments at the beginning of the file.
    buffer_size : int, default 0
        Number of rows to buffer before writing to disk. If 0, write immediately.
    """

    def __init__(self, filename: str,
                 fieldnames: Optional[List[str]] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 buffer_size: int = 0):
        self.filename = filename
        self.file = open(filename, 'w', newline='', encoding='utf-8')
        self.fieldnames = fieldnames
        self.metadata = metadata or {}
        self.buffer_size = buffer_size
        self._buffer: List[Dict[str, Any]] = []
        self._writer = None
        self._header_written = False

        # Write metadata as comments if any
        if self.metadata:
            for key, value in self.metadata.items():
                self.file.write(f"# {key}: {value}\n")

    def _init_writer(self, first_row_keys: List[str]):
        """Initialize the DictWriter and write the header."""
        if self.fieldnames is None:
            self.fieldnames = first_row_keys
        self._writer = csv.DictWriter(self.file, fieldnames=self.fieldnames,
                                      extrasaction='ignore')
        self._writer.writeheader()
        self._header_written = True

    def _flush_buffer(self):
        """Write buffered rows to disk."""
        if not self._buffer:
            return
        if not self._header_written:
            self._init_writer(list(self._buffer[0].keys()))
        for row in self._buffer:
            self._writer.writerow(row)
        self._buffer.clear()

    def write_row(self, data: Dict[str, Any]):
        """
        Write a single row of data.

        Parameters
        ----------
        data : dict
            Dictionary with keys as column names and values as numbers/strings.
        """
        safe_data = {k: self._serialize(v) for k, v in data.items()}

        if self.buffer_size > 0:
            self._buffer.append(safe_data)
            if len(self._buffer) >= self.buffer_size:
                self._flush_buffer()
        else:
            if not self._header_written:
                self._init_writer(list(safe_data.keys()))
            self._writer.writerow(safe_data)

    def write_rows(self, rows: List[Dict[str, Any]]):
        """Write multiple rows."""
        for row in rows:
            self.write_row(row)

    def close(self):
        """Flush buffer and close the CSV file."""
        self._flush_buffer()
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @staticmethod
    def _serialize(value: Any) -> str:
        """
        Convert a value to a string suitable for CSV.

        Floats always keep at least one decimal place:
          0.0  -> '0.0'
          0.1  -> '0.1'
          4.5  -> '4.5'
          1e10 -> '1e+10'  (scientific stays as-is)
        """
        if isinstance(value, (np.floating, float)):
            s = f"{value:g}"
            # :g strips trailing zeros but also removes ".0" for whole numbers.
            # Re-add ".0" when there is no decimal point and no exponent/nan/inf.
            if '.' not in s and 'e' not in s and 'n' not in s and 'i' not in s:
                s += '.0'
            return s
        elif isinstance(value, (np.integer, int)):
            return str(int(value))
        elif isinstance(value, (np.ndarray, list, tuple)):
            return ";".join(str(x) for x in value)
        elif value is None:
            return ""
        else:
            return str(value)


def save_metrics_to_csv(filename: str, timestamps: np.ndarray,
                        metrics_dict: Dict[str, np.ndarray],
                        metadata: Optional[Dict[str, Any]] = None):
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
    metadata : dict, optional
        Global metadata to include as comments.
    """
    with CSVExporter(filename, metadata=metadata) as exporter:
        for i, t in enumerate(timestamps):
            row = {'timestamp': t}
            for name, arr in metrics_dict.items():
                row[name] = arr[i]
            exporter.write_row(row)