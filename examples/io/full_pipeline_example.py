"""
Example: Full pipeline: read video, extract eye landmarks, compute EAR,
and export results to CSV and HDF5.

This demonstrates integration with other EyeTrace modules (eyelids, gaze, etc.).
For simplicity, we simulate landmark detection.
"""

import numpy as np
from eyetrace.io import VideoReader, CSVExporter, HDF5Exporter
from eyetrace.eyelids import ear  # hypothetical EAR function
from eyetrace.gaze import fixation_detector  # hypothetical

def simulate_landmarks(frame):
    """Mock function: return fake eye landmarks for demonstration."""
    h, w = frame.shape[:2]
    # Fake eye bounding boxes
    left_eye = np.array([[w*0.3, h*0.4], [w*0.35, h*0.4], [w*0.35, h*0.45], [w*0.3, h*0.45]])
    right_eye = np.array([[w*0.65, h*0.4], [w*0.7, h*0.4], [w*0.7, h*0.45], [w*0.65, h*0.45]])
    return left_eye, right_eye

def main(video_path="sample.mp4"):
    ear_values = []
    timestamps = []

    with VideoReader(video_path, grayscale=False) as video:
        print(f"Processing video: {video_path}")
        # Prepare CSV exporter (buffered)
        csv_exp = CSVExporter("results.csv", metadata={"video": video_path, "fps": video.fps})
        # Prepare HDF5 exporter (store raw signals)
        h5_exp = HDF5Exporter("results.h5", mode="w")
        h5_exp.create_group("timeseries")

        for frame in video:
            t = video.frame_count / video.fps
            timestamps.append(t)

            # Simulate eye landmark detection
            left_eye, right_eye = simulate_landmarks(frame)

            # Compute EAR (Eye Aspect Ratio)
            ear_left = ear(left_eye)   # hypothetical function
            ear_right = ear(right_eye)
            ear_mean = (ear_left + ear_right) / 2.0
            ear_values.append(ear_mean)

            # Write row to CSV
            row = {
                "timestamp": t,
                "frame": video.frame_count,
                "ear_left": ear_left,
                "ear_right": ear_right,
                "ear_mean": ear_mean
            }
            csv_exp.write_row(row)

        # After loop, write all data to HDF5
        h5_exp.write_array("timeseries/timestamps", np.array(timestamps))
        h5_exp.write_array("timeseries/ear_mean", np.array(ear_values))
        h5_exp.write_attributes("timeseries/ear_mean", {"unit": "ratio", "description": "Eye Aspect Ratio"})
        h5_exp.close()
        csv_exp.close()

        print("Pipeline finished. Results saved to results.csv and results.h5")

if __name__ == "__main__":
    main("path/to/your/video.mp4")