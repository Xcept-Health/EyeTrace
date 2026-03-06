"""
Example: Export metrics to an HDF5 file using HDF5Exporter.

This script creates a structured HDF5 file with datasets and attributes.
"""

import numpy as np
from eyetrace.io import HDF5Exporter

def main():
    # Generate sample data
    timestamps = np.arange(1000) / 50.0  # 50 Hz
    left_pupil = 3.8 + 0.3 * np.random.randn(1000)
    right_pupil = 3.9 + 0.3 * np.random.randn(1000)
    gaze_x = 640 + 200 * np.sin(0.1 * timestamps)
    gaze_y = 480 + 150 * np.cos(0.1 * timestamps)

    # Create HDF5 file
    with HDF5Exporter("eye_data.h5", mode="w") as h5:
        # Store data in groups
        h5.create_group("pupil")
        h5.write_array("pupil/left", left_pupil, compression="gzip")
        h5.write_array("pupil/right", right_pupil, compression="gzip")
        h5.create_group("gaze")
        h5.write_array("gaze/x", gaze_x)
        h5.write_array("gaze/y", gaze_y)
        h5.write_array("timestamps", timestamps)

        # Add metadata as attributes
        h5.write_attributes("/", {
            "subject": "P001",
            "session": "task1",
            "fps": 50,
            "description": "Pupil and gaze data"
        })
        h5.write_attributes("pupil", {
            "unit": "mm",
            "eye": "both"
        })

    print("HDF5 export completed: eye_data.h5")

if __name__ == "__main__":
    main()