"""
Example: Export simulated metrics to a CSV file using CSVExporter.

This script generates synthetic eye-tracking metrics (timestamp, pupil diameter,
blink rate) and saves them to a CSV file with metadata.
"""

import numpy as np
from eyetrace.io import CSVExporter

def main():
    # Simulate some data
    n_frames = 100
    timestamps = np.arange(n_frames) / 30.0  # 30 fps
    pupil_diameter = 4.0 + 0.5 * np.sin(np.linspace(0, 4*np.pi, n_frames)) + 0.1 * np.random.randn(n_frames)
    blink_rate = np.random.poisson(0.2, n_frames)  # random blinks

    # Prepare metadata
    metadata = {
        "subject": "P001",
        "session": "baseline",
        "fps": 30,
        "device": "Tobii Pro"
    }

    # Use CSVExporter with buffering for better performance
    with CSVExporter("output_metrics.csv", metadata=metadata, buffer_size=20) as exporter:
        for i in range(n_frames):
            row = {
                "timestamp": timestamps[i],
                "pupil_diameter_mm": pupil_diameter[i],
                "blink_event": blink_rate[i]
            }
            exporter.write_row(row)

    print("CSV export completed: output_metrics.csv")

if __name__ == "__main__":
    main()