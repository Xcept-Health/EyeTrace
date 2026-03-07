"""
Example: Use Dashboard with vertical layout (video on top, plots below).
Simulates eye tracking metrics on synthetic video.
"""

import numpy as np
import cv2
import time
from eyetrace.io import VideoReader  # not used directly, we use synthetic source
from eyetrace.visualization import Dashboard, draw_text_overlay

# Synthetic video source generator
class SyntheticVideo:
    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_count = 0
        self.start_time = time.time()
    
    def __iter__(self):
        return self
    
    def __next__(self):
        # Create a moving pattern
        t = time.time() - self.start_time
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Moving circle
        x = int(320 + 200 * np.sin(t))
        y = int(240 + 150 * np.cos(t / 2))
        cv2.circle(frame, (x, y), 30, (0, 255, 0), -1)
        
        # Add frame counter
        frame = draw_text_overlay(frame, [f"Frame: {self.frame_count}"], 
                                 position=(10, 30), copy=False)
        
        self.frame_count += 1
        time.sleep(1/self.fps)
        return frame
    
    def __len__(self):
        raise NotImplementedError

def process_frame(frame):
    """Simulate metrics computation."""
    # Simulate realistic metrics
    ear = 0.3 + 0.1 * np.sin(time.time()) + 0.02 * np.random.randn()
    ear = np.clip(ear, 0.2, 0.5)
    
    pupil = 4.0 + 0.5 * np.sin(time.time() / 2) + 0.1 * np.random.randn()
    pupil = np.clip(pupil, 3.0, 5.0)
    
    blink = 1.0 if np.random.rand() < 0.03 else 0.0
    
    # Annotate frame with metrics
    annotated = draw_text_overlay(frame, [
        f"EAR: {ear:.3f}",
        f"Pupil: {pupil:.2f} mm",
        f"Blink: {'YES' if blink > 0 else 'NO'}"
    ], position=(10, 60), copy=False)
    
    return {
        'frame': annotated,
        'timestamp': time.time(),
        'metrics': [ear, pupil, blink]
    }

def main():
    # Create synthetic video source
    video = SyntheticVideo(width=640, height=480, fps=30)
    
    # Plot specifications for three metrics
    plot_specs = [
        {'title': 'Eye Aspect Ratio (EAR)', 'ylabel': 'ratio', 'color': 'b-'},
        {'title': 'Pupil Diameter', 'ylabel': 'mm', 'color': 'g-'},
        {'title': 'Blink Events', 'ylabel': 'binary', 'color': 'r-'}
    ]
    
    # Create dashboard with VERTICAL layout (video on top, plots below)
    with Dashboard(video, plot_specs, update_interval_ms=50,
                   layout='vertical', window_name='EyeTrace - Vertical Layout') as dashboard:
        print("Vertical dashboard running. Press 'q' in the window to quit.")
        print("- Video on top, three plots below")
        print("- All metrics update in real-time")
        dashboard.run(process_frame)

if __name__ == "__main__":
    main()