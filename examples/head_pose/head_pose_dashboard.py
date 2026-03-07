"""
Driver monitoring using head pose.
Tracks head angles, MAR (yawning), nose stability, and postural sag
to detect drowsiness or inattention.
"""

import cv2
import numpy as np
import time
from collections import deque
from eyetrace.io import WebcamReader
from eyetrace.visualization import Dashboard, draw_text_overlay
from eyetrace.head_pose.angles import head_pose_angles
from eyetrace.head_pose.angular_velocity import head_angular_velocity
from eyetrace.head_pose.mar import mouth_aspect_ratio
from eyetrace.head_pose.nose_stability import nose_stability
from eyetrace.head_pose.postural_sag import postural_sag
from eyetrace.head_pose.yawning import yawn_detection, yawn_frequency

# Simulated landmark generator (replace with real MediaPipe in practice)
class SimulatedLandmarks:
    def __init__(self, img_w, img_h, head_pose=(0,0,0), mouth_open=0.0):
        # For demonstration, we create dummy landmarks with given head pose
        # This is highly simplified; real usage would use actual MediaPipe output
        pass

def main():
    # Configuration
    FS = 30  # Hz
    BUFFER_SECONDS = 60
    buffer_len = int(FS * BUFFER_SECONDS)

    # Buffers for metrics
    time_buffer = deque(maxlen=buffer_len)
    pitch_buffer = deque(maxlen=buffer_len)
    roll_buffer = deque(maxlen=buffer_len)
    yaw_buffer = deque(maxlen=buffer_len)
    mar_buffer = deque(maxlen=buffer_len)
    eye_y_buffer = deque(maxlen=buffer_len)  # for postural sag

    # Video source (webcam)
    video = WebcamReader(camera_id=0, width=640, height=480)

    # Plot specifications
    plot_specs = [
        {'title': 'Head Yaw', 'ylabel': 'deg', 'color': 'b-'},
        {'title': 'Head Pitch', 'ylabel': 'deg', 'color': 'g-'},
        {'title': 'Mouth Aspect Ratio', 'ylabel': 'ratio', 'color': 'r-'},
        {'title': 'Nose Stability (variance)', 'ylabel': 'pix²', 'color': 'm-'}
    ]

    def process_frame(frame):
        t = time.time()

        # Simulate head pose and mouth opening based on time (in real app, compute from landmarks)
        # For demonstration, we generate fake values
        yaw = 10 * np.sin(t / 20) + 2 * np.random.randn()
        pitch = 5 * np.sin(t / 15) + 1 * np.random.randn()
        roll = 2 * np.sin(t / 25) + 0.5 * np.random.randn()
        mar = 0.3 + 0.1 * np.sin(t / 10) + 0.05 * np.random.randn()
        if 30 < (t % 60) < 35:  # simulate a yawn every minute
            mar += 0.4
        eye_y = 240 + 5 * np.sin(t / 30) + 2 * np.random.randn()  # eye height

        # Store in buffers
        time_buffer.append(t)
        pitch_buffer.append(pitch)
        roll_buffer.append(roll)
        yaw_buffer.append(yaw)
        mar_buffer.append(mar)
        eye_y_buffer.append(eye_y)

        # Compute derived metrics when enough data
        nose_var = 0.0
        sag_slope = 0.0
        yawn_count = 0
        if len(time_buffer) > 30:
            # Nose stability (simulated: we need nose positions; here we use yaw variance as proxy)
            # In reality, you'd have nose tip coordinates
            nose_var = np.var(list(yaw_buffer)[-30:])  # dummy
            # Postural sag
            eye_y_arr = np.array(eye_y_buffer)
            time_arr = np.array(time_buffer)
            sag_slope = postural_sag(eye_y_arr, time_arr, baseline_seconds=10)
            # Yawn detection
            mar_arr = np.array(mar_buffer)
            yawns = yawn_detection(mar_arr, threshold=0.65, min_duration=2.0, frame_rate=FS)
            yawn_count = len(yawns)

        # Annotate frame
        annotated = frame.copy()
        annotated = draw_text_overlay(annotated, [
            f"Yaw: {yaw:.1f}°",
            f"Pitch: {pitch:.1f}°",
            f"MAR: {mar:.2f}",
            f"Nose var: {nose_var:.1f}",
            f"Sag slope: {sag_slope:.2f} px/s",
            f"Yawns/min: {yawn_count * 60 / max(1, (t - time_buffer[0])/60):.1f}"
        ], position=(10, 30))

        # Alert if drowsy indicators
        if sag_slope > 1.0 or nose_var > 10 or yawn_count > 2:
            cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 120), (0, 0, 255), -1)
            annotated = draw_text_overlay(annotated, [
                "DROWSINESS DETECTED"
            ], position=(10, 50), color=(255, 255, 255))

        return {
            'frame': annotated,
            'timestamp': t,
            'metrics': [yaw, pitch, mar, nose_var]
        }

    # Create dashboard (video + plots)
    with Dashboard(video, plot_specs, update_interval_ms=100,
                   layout='vertical', window_name='Head Pose Monitor') as dashboard:
        print("Head Pose Dashboard running. Press 'q' to quit.")
        dashboard.run(process_frame)

if __name__ == "__main__":
    main()