"""
Simulation: Driver drowsiness detection using eye-tracking metrics.
This example combines:
- Webcam feed (simulated or real)
- Live dashboard with video + plots
- EAR, PERCLOS, pupil diameter simulation
- Drowsiness alert based on metrics
"""

import cv2
import numpy as np
import time
from collections import deque
from eyetrace.io import WebcamReader
from eyetrace.visualization import Dashboard, draw_text_overlay
from eyetrace.eyelids import perclos  # hypothetical, we'll simulate
from eyetrace.signal_analysis.trend import trend_slope

# Simulated functions (replace with real implementations if available)
def simulate_ear(t):
    """Simulate EAR with occasional blinks and drowsy periods."""
    base = 0.3
    # Slow oscillation to simulate drowsiness cycles
    drowsy_cycle = 0.1 * np.sin(t / 30)
    # Random blinks (sharp drops)
    blink = -0.15 if np.random.rand() < 0.02 else 0.0
    # Noise
    noise = 0.02 * np.random.randn()
    ear = base + drowsy_cycle + blink + noise
    return np.clip(ear, 0.1, 0.5)

def simulate_pupil(t):
    """Simulate pupil diameter (mm) with light response and drowsiness."""
    base = 4.0
    # Light changes (simulate day/night? just random slow variation)
    light = 0.5 * np.sin(t / 20)
    # Drowsiness: pupils constrict when drowsy? Actually they tend to fluctuate.
    drowsy = 0.3 * np.sin(t / 10)
    noise = 0.1 * np.random.randn()
    pupil = base + light + drowsy + noise
    return np.clip(pupil, 2.5, 6.0)

def simulate_perclos(ear_history, threshold=0.25):
    """Simulate PERCLOS over a 60-second window."""
    if len(ear_history) < 600:  # assume 10 fps -> 60s
        return 0.0
    # Percentage of frames with EAR < threshold
    closed = sum(1 for e in ear_history if e < threshold)
    return 100.0 * closed / len(ear_history)

def main():
    # Configuration
    ALERT_EAR_THRESHOLD = 0.25
    ALERT_PERCLOS_THRESHOLD = 40.0  # % of time eyes closed over 1 min
    ALERT_PUPIL_VARIANCE_THRESHOLD = 0.3  # high variance may indicate instability

    # Data buffers for trend analysis
    ear_buffer = deque(maxlen=600)      # 60 seconds at 10 fps
    pupil_buffer = deque(maxlen=600)
    timestamps = deque(maxlen=600)

    # Video source (webcam)
    video = WebcamReader(camera_id=0, width=640, height=480)

    # Plot specifications
    plot_specs = [
        {'title': 'Eye Aspect Ratio (EAR)', 'ylabel': 'ratio', 'color': 'b-'},
        {'title': 'Pupil Diameter', 'ylabel': 'mm', 'color': 'g-'},
        {'title': 'PERCLOS (60s)', 'ylabel': '%', 'color': 'r-'}
    ]

    def process_frame(frame):
        """Process each frame: simulate metrics, update buffers, detect drowsiness."""
        t = time.time()

        # Simulate metrics (in real app, compute from landmarks)
        ear = simulate_ear(t)
        pupil = simulate_pupil(t)

        # Update buffers
        ear_buffer.append(ear)
        pupil_buffer.append(pupil)
        timestamps.append(t)

        # Compute PERCLOS over last 60s
        perclos_value = simulate_perclos(list(ear_buffer), threshold=ALERT_EAR_THRESHOLD)

        # Drowsiness detection logic
        alert = False
        alert_reasons = []

        if ear < ALERT_EAR_THRESHOLD:
            alert_reasons.append("Low EAR")
        if perclos_value > ALERT_PERCLOS_THRESHOLD:
            alert_reasons.append("High PERCLOS")
        if len(pupil_buffer) > 100:
            # Check variance of recent pupil data (last 10s)
            recent_pupil = list(pupil_buffer)[-100:]
            if np.var(recent_pupil) > ALERT_PUPIL_VARIANCE_THRESHOLD:
                alert_reasons.append("High pupil variance")
        if alert_reasons:
            alert = True

        # Annotate frame
        annotated = frame.copy()
        annotated = draw_text_overlay(annotated, [
            f"EAR: {ear:.3f}",
            f"Pupil: {pupil:.2f} mm",
            f"PERCLOS: {perclos_value:.1f}%"
        ], position=(10, 30), color=(255, 255, 255))

        if alert:
            # Red background for alert
            cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 80), (0, 0, 255), -1)
            annotated = draw_text_overlay(annotated, [
                " DROWSINESS ALERT ",
                f"Reasons: {', '.join(alert_reasons)}"
            ], position=(10, 30), color=(255, 255, 255), line_spacing=30)

        return {
            'frame': annotated,
            'timestamp': t,
            'metrics': [ear, pupil, perclos_value]
        }

    # Create dashboard with vertical layout (video on top, plots below)
    with Dashboard(video, plot_specs, update_interval_ms=100,
                   layout='vertical', window_name='Drowsiness Detector') as dashboard:
        print(" Driver Drowsiness Detection Dashboard")
        print("Press 'q' in the window to quit.")
        dashboard.run(process_frame)

if __name__ == "__main__":
    main()