"""
Example: Use the visualization module to display live video and metrics.

This script simulates a video source (random images) and computes fake metrics
(pupil diameter, blink rate) to demonstrate the dashboard.
"""

import numpy as np
import cv2
import time
from eyetrace.io import VideoReader  # or use a real video
from eyetrace.visualization import Dashboard, draw_text_overlay

# If you don't have a real video, create a synthetic source
class SyntheticVideo:
    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_count = 0

    def __iter__(self):
        return self

    def __next__(self):
        # Generate a random frame (or a pattern)
        frame = np.random.randint(0, 255, (self.height, self.width, 3), dtype=np.uint8)
        # Add some text to show frame number
        frame = draw_text_overlay(frame, [f"Frame: {self.frame_count}"], copy=False)
        self.frame_count += 1
        time.sleep(1/self.fps)  # simulate real-time
        return frame

    def __len__(self):
        raise NotImplementedError

# Simulated metrics computation
def process_frame(frame):
    """User-defined processing function."""
    # Simulate pupil diameter and blink rate
    pupil = 4.0 + 0.5 * np.sin(time.time() / 2) + 0.1 * np.random.randn()
    blink = 1.0 if np.random.rand() < 0.05 else 0.0  # random blinks

    # Annotate frame with metrics
    annotated = draw_text_overlay(frame, [
        f"Pupil: {pupil:.2f} mm",
        f"Blink: {'YES' if blink > 0 else 'NO'}"
    ], position=(10, 60))

    return {
        'frame': annotated,
        'timestamp': time.time(),
        'metrics': [pupil, blink]
    }

def main():
    # Use synthetic video source (replace with VideoReader for real files)
    video = SyntheticVideo()

    # Define plots
    plot_specs = [
        {'title': 'Pupil Diameter', 'ylabel': 'mm', 'color': 'b-'},
        {'title': 'Blink Events', 'ylabel': 'binary', 'color': 'r-'}
    ]

    # Create and run dashboard
    with Dashboard(video, plot_specs, update_interval_ms=100) as dashboard:
        print("Dashboard running. Press 'q' in video window to quit.")
        dashboard.run(process_frame)

if __name__ == "__main__":
    main()