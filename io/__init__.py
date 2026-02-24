"""
Input/Output module for EyeTrace.

Provides classes and functions to read video sources (webcam, files) and
export metrics to various formats (CSV, HDF5).
"""

from .video import VideoReader, WebcamReader
from .csv_exporter import CSVExporter
from .hdf5_exporter import HDF5Exporter

__all__ = [
    'VideoReader',
    'WebcamReader',
    'CSVExporter',
    'HDF5Exporter',
]