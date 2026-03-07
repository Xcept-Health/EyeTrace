#!/usr/bin/env python
"""
Concussion screening using gaze stability: saccade frequency, fixation dispersion, and gaze entropy.
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
    from eyetrace.gaze.saccades import detect_saccades, saccade_fixation_ratio
    from eyetrace.gaze.fixation import fixation_dispersion, gaze_centroid
    from eyetrace.gaze.entropy import gaze_entropy
    from eyetrace.gaze.utils import angular_velocity
except ImportError:
    print("Error: EyeTrace must be installed.")
    sys.exit(1)


# Configuration

RECORDING_DURATION = 30
FS = 30
SACCADE_FREQ_THRESH = 2.0          # Hz
DISPERSION_THRESH = 20              # pixels
ENTROPY_THRESH = 4.0                # bits

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
GRAPH_HEIGHT = 120

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# For gaze, we approximate using the midpoint between the two irises
LEFT_IRIS = list(range(468, 473))
RIGHT_IRIS = list(range(473, 478))

def get_iris_center(face_landmarks, indices, w, h):
    pts = []
    for idx in indices:
        lm = face_landmarks.landmark[idx]
        pts.append([lm.x * w, lm.y * h])
    return np.mean(pts, axis=0)

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

    print("=== Concussion Screener (Gaze Stability) ===")
    print("Look at the white dot and keep your head still.")
    print("Press SPACE to start recording, 'q' to quit.")

    # Live buffers
    buf_x = deque(maxlen=150)
    buf_y = deque(maxlen=150)

    # Data storage
    timestamps = []
    gaze_x = []
    gaze_y = []

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

        gaze_pt = None
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            left_center = get_iris_center(face_landmarks, LEFT_IRIS, w_frame, h_frame)
            right_center = get_iris_center(face_landmarks, RIGHT_IRIS, w_frame, h_frame)
            gaze_pt = (left_center + right_center) / 2.0

            # Draw iris centers
            cv2.circle(display, tuple(left_center.astype(int)), 3, (255,255,0), -1)
            cv2.circle(display, tuple(right_center.astype(int)), 3, (0,255,255), -1)

            if recording:
                timestamps.append(current_time)
                gaze_x.append(gaze_pt[0])
                gaze_y.append(gaze_pt[1])

            # Update live buffers
            buf_x.append(gaze_pt[0])
            buf_y.append(gaze_pt[1])

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
        draw_graph(display, list(buf_x), (255,100,100), graph_y, GRAPH_HEIGHT, "Gaze X")
        draw_graph(display, list(buf_y), (100,255,255), graph_y+GRAPH_HEIGHT+10, GRAPH_HEIGHT, "Gaze Y")

        cv2.imshow("Concussion Screener", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' ') and not recording:
            recording = True
            start_time = time.time()
            timestamps.clear()
            gaze_x.clear()
            gaze_y.clear()
            print("Recording started...")

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()

    if len(timestamps) < 30:
        print("Insufficient data.")
        return

    t = np.array(timestamps) - timestamps[0]
    gx = np.array(gaze_x)
    gy = np.array(gaze_y)
    gaze_pos = np.column_stack((gx, gy))

    # Saccade detection (need 3D vectors: assume z=0)
    gaze_3d = np.zeros((len(t), 3))
    gaze_3d[:,0] = gx
    gaze_3d[:,1] = gy
    saccades = detect_saccades(gaze_3d, t, velocity_threshold=1.0, min_duration=0.02)
    saccade_freq = len(saccades) / t[-1] if t[-1] > 0 else 0

    # Fixation dispersion
    # Simple: assume fixations are periods not in saccades. We'll compute overall dispersion as mean of per-point dispersion?
    # For demo, use the dispersion of all points (simplified)
    disp = np.mean(np.std(gaze_pos, axis=0))

    # Gaze entropy
    entropy = gaze_entropy(gaze_pos, bins=15, range_x=(0, w_frame), range_y=(0, h_frame))

    print("\n--- Gaze Stability Results ---")
    print(f"Microsaccade frequency: {saccade_freq:.2f} Hz")
    print(f"Overall dispersion: {disp:.2f} pixels")
    print(f"Gaze entropy: {entropy:.2f} bits")

    risk = 0
    if saccade_freq > SACCADE_FREQ_THRESH:
        risk += 1
    if disp > DISPERSION_THRESH:
        risk += 1
    if entropy < ENTROPY_THRESH:   # low entropy = too focused
        risk += 1
    risk_level = ["Low", "Moderate", "High", "Very High"][min(risk,3)]

    print(f"\nConcussion risk: {risk_level} ({risk}/3)")

    with open("concussion_screening_result.txt", "w") as f:
        f.write(f"Recording duration: {t[-1]:.2f} s\n")
        f.write(f"Microsaccade frequency: {saccade_freq:.2f} Hz\n")
        f.write(f"Overall dispersion: {disp:.2f} px\n")
        f.write(f"Gaze entropy: {entropy:.2f} bits\n")
        f.write(f"Risk score: {risk}/3 - {risk_level}\n")

    print("\nReport saved to 'concussion_screening_result.txt'.")

if __name__ == "__main__":
    main()