#!/usr/bin/env python
"""
Cervical dystonia / torticollis screener using head pose angles (yaw, pitch, roll).
Measures asymmetry, range of motion, and stability.
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from eyetrace.head_pose.angles import head_pose_angles
    from eyetrace.head_pose.angular_velocity import head_angular_velocity
    from eyetrace.head_pose.nose_stability import nose_stability
    from eyetrace.head_pose.postural_sag import postural_sag
except ImportError:
    print("Error: EyeTrace must be installed.")
    sys.exit(1)

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
RECORDING_DURATION = 30
FS = 30
YAW_ASYMMETRY_THRESH = 10          # degrees
PITCH_RANGE_THRESH = 5             # degrees (min range for normal)
STABILITY_THRESH = 50               # pixel variance

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
GRAPH_HEIGHT = 120

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def draw_graph(img, data, color, y_offset, height, label):
    if len(data) < 2:
        cv2.putText(img, f"{label}: no data", (10, y_offset+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return
    arr = np.array(data)
    min_val, max_val = np.min(arr), np.max(arr)
    if max_val - min_val < 1e-6:
        scale = 1.0
    else:
        scale = height / (max_val - min_val)
    pts = []
    x_step = img.shape[1] / (len(data)-1)
    for i, val in enumerate(data):
        x = int(i * x_step)
        y = int(y_offset + height - (val - min_val) * scale)
        pts.append([x, y])
    pts = np.array(pts, dtype=np.int32)
    cv2.polylines(img, [pts], False, color, 2)
    cv2.putText(img, f"{label}: {data[-1]:.2f}", (10, y_offset+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot access webcam.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT - GRAPH_HEIGHT)

    print("=== Cervical Dystonia / Torticollis Screener (Head Pose) ===")
    print("Look straight ahead. Try to keep your head still.")
    print("Press SPACE to start recording, 'q' to quit.")

    # Live buffers
    buf_yaw = deque(maxlen=150)
    buf_pitch = deque(maxlen=150)
    buf_roll = deque(maxlen=150)

    # Data storage
    timestamps = []
    yaw_vals = []
    pitch_vals = []
    roll_vals = []

    recording = False
    start_time = None

    prev_time = time.time()
    fps = 0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h_frame, w_frame, _ = frame.shape

        display = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
        display[h_frame:, :] = (40,40,40)
        display[:h_frame, :w_frame] = frame

        cv2.circle(display, (w_frame//2, h_frame//2), 5, (255,255,255), -1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        current_time = time.time()
        dt = current_time - prev_time
        frame_count += 1
        if frame_count >= 10:
            fps = 10 / dt if dt>0 else 0
            prev_time = current_time
            frame_count = 0

        angles = None
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            pitch, roll, yaw = head_pose_angles(face_landmarks, w_frame, h_frame)
            angles = (pitch, roll, yaw)

            # Convert to degrees for display
            pitch_deg = np.degrees(pitch)
            roll_deg = np.degrees(roll)
            yaw_deg = np.degrees(yaw)

            # Draw a simple line indicating head orientation (optional)
            # Here just display text

            if recording:
                timestamps.append(current_time)
                yaw_vals.append(yaw_deg)
                pitch_vals.append(pitch_deg)
                roll_vals.append(roll_deg)

            # Update live buffers
            buf_yaw.append(yaw_deg)
            buf_pitch.append(pitch_deg)
            buf_roll.append(roll_deg)

        # UI
        if not recording:
            cv2.putText(display, "Press SPACE to start recording", (30,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        else:
            elapsed = current_time - start_time
            remaining = max(0, RECORDING_DURATION - elapsed)
            cv2.putText(display, f"Recording... {remaining:.1f}s", (30,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            if elapsed >= RECORDING_DURATION:
                break

        cv2.putText(display, f"FPS: {fps:.1f}", (WINDOW_WIDTH-100,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # Draw live graphs
        graph_y = h_frame + 10
        draw_graph(display, list(buf_yaw), (255,100,100), graph_y, GRAPH_HEIGHT, "Yaw (deg)")
        draw_graph(display, list(buf_pitch), (100,255,100), graph_y+GRAPH_HEIGHT+10, GRAPH_HEIGHT, "Pitch (deg)")
        draw_graph(display, list(buf_roll), (100,100,255), graph_y+2*(GRAPH_HEIGHT+10), GRAPH_HEIGHT, "Roll (deg)")

        cv2.imshow("Cervical Dystonia Screener", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' ') and not recording:
            recording = True
            start_time = time.time()
            timestamps.clear()
            yaw_vals.clear()
            pitch_vals.clear()
            roll_vals.clear()
            print("Recording started...")

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()

    if len(timestamps) < 30:
        print("Insufficient data.")
        return

    t = np.array(timestamps) - timestamps[0]
    yaw = np.array(yaw_vals)
    pitch = np.array(pitch_vals)
    roll = np.array(roll_vals)

    # Asymmetry: mean yaw offset from zero
    mean_yaw = np.mean(yaw)
    mean_pitch = np.mean(pitch)
    mean_roll = np.mean(roll)

    # Range of motion (max-min)
    yaw_range = np.max(yaw) - np.min(yaw)
    pitch_range = np.max(pitch) - np.min(pitch)
    roll_range = np.max(roll) - np.min(roll)

    # Stability: variance
    yaw_var = np.var(yaw)
    pitch_var = np.var(pitch)
    roll_var = np.var(roll)

    print("\n--- Head Pose Analysis ---")
    print(f"Mean yaw: {mean_yaw:.2f}° (asymmetry)")
    print(f"Mean pitch: {mean_pitch:.2f}°")
    print(f"Mean roll: {mean_roll:.2f}°")
    print(f"Yaw range: {yaw_range:.2f}°")
    print(f"Pitch range: {pitch_range:.2f}°")
    print(f"Roll range: {roll_range:.2f}°")
    print(f"Yaw variance: {yaw_var:.2f}")
    print(f"Pitch variance: {pitch_var:.2f}")
    print(f"Roll variance: {roll_var:.2f}")

    # Simple risk scoring
    risk = 0
    if abs(mean_yaw) > YAW_ASYMMETRY_THRESH:
        risk += 1
    if pitch_range < PITCH_RANGE_THRESH:
        risk += 1
    if yaw_var > STABILITY_THRESH:
        risk += 1
    risk_level = ["Low", "Moderate", "High"][min(risk,2)]

    print(f"\nCervical dystonia risk: {risk_level} ({risk}/3)")

    with open("cervical_dystonia_screening_result.txt", "w") as f:
        f.write(f"Recording duration: {t[-1]:.2f} s\n")
        f.write(f"Mean yaw: {mean_yaw:.2f}°\n")
        f.write(f"Mean pitch: {mean_pitch:.2f}°\n")
        f.write(f"Mean roll: {mean_roll:.2f}°\n")
        f.write(f"Yaw range: {yaw_range:.2f}°\n")
        f.write(f"Pitch range: {pitch_range:.2f}°\n")
        f.write(f"Roll range: {roll_range:.2f}°\n")
        f.write(f"Yaw variance: {yaw_var:.2f}\n")
        f.write(f"Risk score: {risk}/3 - {risk_level}\n")

    print("\nReport saved to 'cervical_dystonia_screening_result.txt'.")

if __name__ == "__main__":
    main()