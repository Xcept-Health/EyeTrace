"""
Pupil dynamics monitoring with live dashboard.
Simulates a person performing a task; we track pupil diameter,
velocity, and hippus amplitude to detect signs of fatigue or cognitive load.
"""

import cv2
import numpy as np
import time
from collections import deque
from eyetrace.io import WebcamReader
from eyetrace.visualization import Dashboard, draw_text_overlay
from eyetrace.pupil.dynamics import first_derivative, hippus_amplitude
from eyetrace.pupil.metrics import variance, coefficient_variation

# Simulated pupil signal (replace with real extraction if available)
def simulate_pupil_diameter(t, fatigue_level=0.0):
    """
    Simulate pupil diameter with:
    - baseline 4.0 mm
    - slow drift (fatigue causes dilation?)
    - hippus oscillations (1-2 Hz)
    - occasional constrictions (light changes)
    """
    baseline = 4.0
    # Fatigue tends to increase baseline and variability
    fatigue_effect = fatigue_level * 0.5  # up to +0.5 mm
    # Hippus
    hippus = 0.2 * np.sin(2 * np.pi * 1.5 * t) + 0.1 * np.sin(2 * np.pi * 0.8 * t)
    # Random light reflexes (simulate)
    if np.random.rand() < 0.005:  # occasional event
        constriction = -0.8 * np.exp(-((t % 10) - 0.2)**2 / 0.01)
    else:
        constriction = 0.0
    # Noise
    noise = 0.05 * np.random.randn()
    return baseline + fatigue_effect + hippus + constriction + noise

def main():
    # Configuration
    FS = 30  # Hz (simulated)
    FATIGUE_INCREASE_RATE = 0.01  # per second

    # Buffers for real-time analysis
    maxlen = 300  # 10 seconds at 30 Hz
    dia_buffer = deque(maxlen=maxlen)
    time_buffer = deque(maxlen=maxlen)
    hippus_buffer = deque(maxlen=60)  # 2 seconds for hippus calc

    # Video source (webcam)
    video = WebcamReader(camera_id=0, width=640, height=480)

    # Fatigue level (simulated)
    fatigue = 0.0

    # Plot specifications
    plot_specs = [
        {'title': 'Pupil Diameter', 'ylabel': 'mm', 'color': 'b-'},
        {'title': 'Velocity', 'ylabel': 'mm/s', 'color': 'g-'},
        {'title': 'Hippus Amplitude', 'ylabel': 'mm', 'color': 'r-'}
    ]

    def process_frame(frame):
        nonlocal fatigue
        t = time.time()

        # Simulate pupil diameter (in real app, extract from image)
        dia = simulate_pupil_diameter(t, fatigue)

        # Update buffers
        dia_buffer.append(dia)
        time_buffer.append(t)

        # Increase fatigue slowly (simulate time-on-task)
        fatigue += FATIGUE_INCREASE_RATE / FS
        fatigue = min(fatigue, 1.0)  # cap at 1.0

        # Compute metrics
        if len(dia_buffer) > 5:
            # Velocity
            vel = first_derivative(np.array(dia_buffer), np.array(time_buffer), smooth=True)
            current_vel = vel[-1] if len(vel) > 0 else 0.0

            # Hippus amplitude over last 2 seconds
            if len(dia_buffer) >= 60:
                recent_dia = np.array(dia_buffer)[-60:]
                hippus = hippus_amplitude(recent_dia, FS, lowcut=0.5, highcut=4.0, method='rms')
                hippus_buffer.append(hippus)
            else:
                hippus = 0.0
                hippus_buffer.append(0.0)

            # Variability over last 10s
            if len(dia_buffer) >= 300:
                cv = coefficient_variation(np.array(dia_buffer))
            else:
                cv = 0.0
        else:
            current_vel = 0.0
            hippus = 0.0
            cv = 0.0

        # Annotate frame
        annotated = frame.copy()
        annotated = draw_text_overlay(annotated, [
            f"Pupil: {dia:.2f} mm",
            f"Velocity: {current_vel:.2f} mm/s",
            f"Hippus: {hippus:.3f} mm",
            f"CV (10s): {cv:.2f}%",
            f"Fatigue: {fatigue:.2f}"
        ], position=(10, 30))

        # Alert if high fatigue or abnormal metrics
        if fatigue > 0.8 or cv > 10.0 or hippus > 0.5:
            cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 90), (0, 0, 255), -1)
            annotated = draw_text_overlay(annotated, [
                "  FATIGUE / HIGH VARIABILITY  "
            ], position=(10, 50), color=(255, 255, 255))

        return {
            'frame': annotated,
            'timestamp': t,
            'metrics': [dia, current_vel, hippus]
        }

    # Create dashboard (video + plots)
    with Dashboard(video, plot_specs, update_interval_ms=100,
                   layout='vertical', window_name='Pupil Dynamics Monitor') as dashboard:
        print("Pupil Dynamics Dashboard running. Press 'q' to quit.")
        dashboard.run(process_frame)

if __name__ == "__main__":
    main()