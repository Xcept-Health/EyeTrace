#!/usr/bin/env python
"""
Screening for autonomic dysfunction using pupillary light reflex (PLR) and hippus.
Measures: baseline diameter, constriction latency, constriction amplitude, hippus amplitude.
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
    from eyetrace.pupil.core import extract_pupil_diameter
    from eyetrace.pupil.dynamics import hippus_amplitude
    from eyetrace.pupil.plr import plr_analysis
except ImportError:
    print("Error: EyeTrace must be installed. Run 'pip install -e .' from project root.")
    sys.exit(1)


# Configuration

RECORDING_DURATION = 30          # seconds
LIGHT_STIMULUS_TIME = 10.0        # time when light stimulus is applied (user should blink or something)
FS = 30                           # approx webcam fps
HIPPUS_THRESH = 0.5               # mm (adjust based on calibration)
LATENCY_THRESH = 0.3              # seconds

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
GRAPH_HEIGHT = 120

# MediaPipe initialization
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_IRIS = list(range(468, 473))
RIGHT_IRIS = list(range(473, 478))

def get_iris_landmarks(face_landmarks, indices, w, h):
    points = []
    for idx in indices:
        lm = face_landmarks.landmark[idx]
        x = int(lm.x * w)
        y = int(lm.y * h)
        points.append([x, y])
    return np.array(points, dtype=np.float64)

def draw_graph(img, data, color, y_offset, height, label):
    if len(data) < 2:
        cv2.putText(img, f"{label}: no data", (10, y_offset + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return
    arr = np.array(data)
    min_val = np.min(arr)
    max_val = np.max(arr)
    if max_val - min_val < 1e-6:
        scale = 1.0
    else:
        scale = height / (max_val - min_val)
    pts = []
    x_step = img.shape[1] / (len(data) - 1)
    for i, val in enumerate(data):
        x = int(i * x_step)
        y = int(y_offset + height - (val - min_val) * scale)
        pts.append([x, y])
    pts = np.array(pts, dtype=np.int32)
    cv2.polylines(img, [pts], False, color, 2)
    cv2.putText(img, f"{label}: {data[-1]:.2f}", (10, y_offset + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot access webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT - GRAPH_HEIGHT)

    print("=== Autonomic Dysfunction Screener (Pupil) ===")
    print("Look at the white dot. A light stimulus will be simulated at 10s.")
    print("Press SPACE to start recording, 'q' to quit.")

    # Buffers for live display
    buf_diam = deque(maxlen=150)  # last 5 seconds at 30 fps

    # Data storage
    timestamps = []
    diameters = []

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

        # Create display canvas
        display = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
        display[h_frame:, :] = (40, 40, 40)
        display[:h_frame, :w_frame] = frame

        # Draw fixation dot
        cv2.circle(display, (w_frame//2, h_frame//2), 5, (255,255,255), -1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        current_time = time.time()
        dt = current_time - prev_time
        frame_count += 1
        if frame_count >= 10:
            fps = 10 / dt if dt > 0 else 0
            prev_time = current_time
            frame_count = 0

        diam = None
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            left_iris = get_iris_landmarks(face_landmarks, LEFT_IRIS, w_frame, h_frame)
            right_iris = get_iris_landmarks(face_landmarks, RIGHT_IRIS, w_frame, h_frame)

            d_left = extract_pupil_diameter(left_iris, w_frame, h_frame)
            d_right = extract_pupil_diameter(right_iris, w_frame, h_frame)
            diam = (d_left + d_right) / 2.0

            # Draw iris landmarks
            for pt in left_iris:
                cv2.circle(display, tuple(pt.astype(int)), 2, (255,255,0), -1)
            for pt in right_iris:
                cv2.circle(display, tuple(pt.astype(int)), 2, (0,255,255), -1)

            if recording:
                timestamps.append(current_time)
                diameters.append(diam)

            # Update live buffer
            if diam is not None:
                buf_diam.append(diam)

        # UI
        if not recording:
            cv2.putText(display, "Press SPACE to start recording", (30,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        else:
            elapsed = current_time - start_time
            remaining = max(0, RECORDING_DURATION - elapsed)
            cv2.putText(display, f"Recording... {remaining:.1f}s", (30,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            if elapsed >= RECORDING_DURATION:
                break

        # FPS
        cv2.putText(display, f"FPS: {fps:.1f}", (WINDOW_WIDTH-100,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # Draw live graph
        graph_y = h_frame + 10
        draw_graph(display, list(buf_diam), (0,255,0), graph_y, GRAPH_HEIGHT, "Diameter (px)")

        cv2.imshow("Autonomic Dysfunction Screener", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' ') and not recording:
            recording = True
            start_time = time.time()
            timestamps.clear()
            diameters.clear()
            print("Recording started...")

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()

    if len(timestamps) < 30:
        print("Insufficient data.")
        return

    # Analysis
    t = np.array(timestamps) - timestamps[0]
    d = np.array(diameters)

    # Simulate light stimulus at LIGHT_STIMULUS_TIME (in real app, you would mark it)
    # For demo, we assume stimulus occurred at that time relative to start
    if t[-1] > LIGHT_STIMULUS_TIME:
        plr = plr_analysis(d, t, stimulus_time=LIGHT_STIMULUS_TIME,
                           baseline_duration=2.0, response_window=3.0)
    else:
        plr = {'latency': np.nan, 'amplitude': np.nan}

    hippus = hippus_amplitude(d, fs=FS, method='rms')

    print("\n--- Pupil Analysis Results ---")
    print(f"Mean baseline diameter: {np.mean(d[t<LIGHT_STIMULUS_TIME]):.2f} px")
    print(f"PLR latency: {plr.get('latency', np.nan):.3f} s")
    print(f"PLR amplitude: {plr.get('amplitude', np.nan):.2f} px")
    print(f"Hippus amplitude (RMS): {hippus:.3f} px")

    # Simple risk score
    risk = 0
    if plr.get('latency', np.inf) > LATENCY_THRESH:
        risk += 1
    if hippus > HIPPUS_THRESH:
        risk += 1
    risk_level = ["Low", "Moderate", "High"][min(risk,2)]

    print(f"\nAutonomic dysfunction risk: {risk_level} ({risk}/2)")

    with open("autonomic_screening_result.txt", "w") as f:
        f.write(f"Recording duration: {t[-1]:.2f} s\n")
        f.write(f"Mean diameter: {np.mean(d):.2f} px\n")
        f.write(f"PLR latency: {plr.get('latency', np.nan):.3f} s\n")
        f.write(f"PLR amplitude: {plr.get('amplitude', np.nan):.2f} px\n")
        f.write(f"Hippus amplitude: {hippus:.3f} px\n")
        f.write(f"Risk score: {risk}/2 - {risk_level}\n")

    print("\nReport saved to 'autonomic_screening_result.txt'.")

if __name__ == "__main__":
    main()