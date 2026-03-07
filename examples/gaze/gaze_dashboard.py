"""
Gaze analysis dashboard.
Simulates a person looking at a screen; detects fixations and saccades,
displays gaze entropy over time, and alerts if unusual patterns (e.g., staring too long).
"""

import cv2
import numpy as np
import time
from collections import deque
from eyetrace.io import WebcamReader
from eyetrace.visualization import Dashboard, draw_text_overlay
from eyetrace.gaze.entropy import gaze_entropy
from eyetrace.gaze.fixation import fixation_duration, fixation_dispersion, gaze_centroid
from eyetrace.gaze.saccades import detect_saccades, saccade_fixation_ratio

def simulate_gaze_position(t):
    """
    Simulate gaze on screen (1920x1080) with:
    - smooth pursuit (slow oscillation)
    - occasional saccades
    - fixations
    """
    base_x = 960 + 300 * np.sin(0.2 * t)
    base_y = 540 + 200 * np.cos(0.15 * t)
    # Saccade every ~10s
    if (t % 10) < 0.2:
        # Rapid jump
        saccade_x = np.random.choice([-200, 200]) if np.random.rand() > 0.5 else 0
        saccade_y = np.random.choice([-150, 150]) if np.random.rand() > 0.5 else 0
    else:
        saccade_x, saccade_y = 0, 0
    # Noise
    noise_x = np.random.randn() * 10
    noise_y = np.random.randn() * 10
    x = base_x + saccade_x + noise_x
    y = base_y + saccade_y + noise_y
    return np.clip(x, 0, 1920), np.clip(y, 0, 1080)

def main():
    # Configuration
    FS = 30  # Hz (simulated)
    BUFFER_SECONDS = 30
    buffer_len = int(FS * BUFFER_SECONDS)

    # Buffers
    time_buffer = deque(maxlen=buffer_len)
    gaze_x_buffer = deque(maxlen=buffer_len)
    gaze_y_buffer = deque(maxlen=buffer_len)

    # Video source (webcam)
    video = WebcamReader(camera_id=0, width=640, height=480)

    # Plot specifications
    plot_specs = [
        {'title': 'Gaze X', 'ylabel': 'pixels', 'color': 'b-'},
        {'title': 'Gaze Y', 'ylabel': 'pixels', 'color': 'g-'},
        {'title': 'Gaze Entropy (30s)', 'ylabel': 'bits', 'color': 'r-'},
        {'title': 'Fixation Dispersion', 'ylabel': 'pixels', 'color': 'm-'}
    ]

    def process_frame(frame):
        t = time.time()

        # Simulate gaze position
        x, y = simulate_gaze_position(t)

        # Store in buffers
        time_buffer.append(t)
        gaze_x_buffer.append(x)
        gaze_y_buffer.append(y)

        # Compute derived metrics when enough data
        entropy = 0.0
        fixation_disp = 0.0
        saccade_ratio = 0.0
        centroid = np.array([np.nan, np.nan])

        if len(time_buffer) > 50:
            # Build arrays
            times = np.array(time_buffer)
            gaze_positions = np.column_stack((gaze_x_buffer, gaze_y_buffer))

            # Entropy over last 30s
            entropy = gaze_entropy(gaze_positions, bins=15,
                                   range_x=(0,1920), range_y=(0,1080))

            # Simple fixation detection: assume fixations when velocity < threshold
            dt = times[1:] - times[:-1]
            dx = np.diff(gaze_x_buffer)
            dy = np.diff(gaze_y_buffer)
            speed = np.sqrt(dx**2 + dy**2) / dt
            speed = np.append(speed, speed[-1])  # same length
            fixation_mask = speed < 30  # pixels/s threshold

            # Fixation dispersion
            if np.any(fixation_mask):
                disp = fixation_dispersion(gaze_positions, fixation_mask)
                if disp:
                    fixation_disp = np.mean(disp)
                centroid = gaze_centroid(gaze_positions, fixation_mask)

            # Saccade detection (simplified)
            # For demo, we just use a threshold-based detection
            from scipy.ndimage import label
            is_saccade = speed > 100
            labeled, n_sacc = label(is_saccade)
            saccades = []
            for i in range(1, n_sacc+1):
                region = np.where(labeled == i)[0]
                if len(region) > 3:
                    saccades.append((region[0], region[-1]))
            if len(times) > 0:
                saccade_ratio = saccade_fixation_ratio(saccades, len(times))

        # Annotate frame
        annotated = frame.copy()
        annotated = draw_text_overlay(annotated, [
            f"Gaze: ({int(x)}, {int(y)})",
            f"Entropy: {entropy:.2f} bits",
            f"Fixation disp: {fixation_disp:.1f}",
            f"Saccade ratio: {saccade_ratio:.2f}",
            f"Centroid: ({int(centroid[0]) if not np.isnan(centroid[0]) else 0}, "
            f"{int(centroid[1]) if not np.isnan(centroid[1]) else 0})"
        ], position=(10, 30))

        # Alert if abnormal
        if entropy < 1.5 or fixation_disp < 5:
            cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 120), (0, 0, 255), -1)
            annotated = draw_text_overlay(annotated, [
                "UNUSUAL GAZE PATTERN "
            ], position=(10, 50), color=(255, 255, 255))

        return {
            'frame': annotated,
            'timestamp': t,
            'metrics': [x, y, entropy, fixation_disp]
        }

    # Create dashboard (video + plots)
    with Dashboard(video, plot_specs, update_interval_ms=100,
                   layout='vertical', window_name='Gaze Analysis Dashboard') as dashboard:
        print("Gaze Dashboard running. Press 'q' to quit.")
        dashboard.run(process_frame)

if __name__ == "__main__":
    main()