#!/usr/bin/env python
"""
Fatigue / neurological screener using eyelid metrics: blink duration, closing/opening speeds, PERCLOS.
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
    from eyetrace.eyelids.utils import extract_both_eyes
    from eyetrace.eyelids.ear import both_eyes_ear
    from eyetrace.eyelids.blink_detection import detect_blinks, blink_frequency, mean_closure_duration
    from eyetrace.eyelids.eyelid_speed import eyelid_closing_speed, eyelid_opening_speed
    from eyetrace.eyelids.perclos import perclos
except ImportError:
    print("Error: EyeTrace must be installed.")
    sys.exit(1)


# Configuration

RECORDING_DURATION = 30
FS = 30
EAR_THRESH = 0.2
BLINK_DURATION_THRESH = 0.3       # seconds (long blink indicator)
CLOSING_SPEED_THRESH = -5.0        # EAR/s (more negative = faster closing)
PERCLOS_THRESH = 15                # percent

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

    print("=== Fatigue / Neurological Screener (Eyelids) ===")
    print("Look at the white dot. Try to blink naturally.")
    print("Press SPACE to start recording, 'q' to quit.")

    # Live buffer
    buf_ear = deque(maxlen=150)

    # Data storage
    timestamps = []
    ear_values = []

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

        ear = None
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            left_eye, right_eye = extract_both_eyes(face_landmarks, w_frame, h_frame)
            ear = both_eyes_ear(left_eye, right_eye)

            # Draw eye landmarks
            for pt in left_eye.astype(int):
                cv2.circle(display, tuple(pt), 2, (255,255,0), -1)
            for pt in right_eye.astype(int):
                cv2.circle(display, tuple(pt), 2, (0,255,255), -1)

            if recording:
                timestamps.append(current_time)
                ear_values.append(ear)

            # Update live buffer
            buf_ear.append(ear)

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

        # Draw live EAR graph
        graph_y = h_frame + 10
        draw_graph(display, list(buf_ear), (0,255,0), graph_y, GRAPH_HEIGHT, "EAR")

        cv2.imshow("Fatigue Screener", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' ') and not recording:
            recording = True
            start_time = time.time()
            timestamps.clear()
            ear_values.clear()
            print("Recording started...")

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()

    if len(timestamps) < 30:
        print("Insufficient data.")
        return

    t = np.array(timestamps) - timestamps[0]
    ear = np.array(ear_values)

    # Blink detection
    blinks = detect_blinks(ear, threshold=EAR_THRESH, min_interval_frames=10)
    blink_freq = blink_frequency(blinks, t[-1]) if t[-1] > 0 else 0
    mean_dur = mean_closure_duration(blinks, frame_rate=FS) if blinks else 0

    # Eyelid speeds
    closing_speeds = eyelid_closing_speed(ear, t, threshold=EAR_THRESH, smooth=True)
    opening_speeds = eyelid_opening_speed(ear, t, threshold=EAR_THRESH, smooth=True)

    # PERCLOS over last 30 seconds
    perc = perclos(ear, threshold=EAR_THRESH, window_seconds=30, frame_rate=FS)

    print("\n--- Eyelid Dynamics Results ---")
    print(f"Number of blinks: {len(blinks)}")
    print(f"Blink frequency: {blink_freq:.2f} blinks/min")
    print(f"Mean blink duration: {mean_dur*1000:.1f} ms")
    if len(closing_speeds) > 0:
        print(f"Mean closing speed: {np.mean(closing_speeds):.2f} EAR/s")
        print(f"Mean opening speed: {np.mean(opening_speeds):.2f} EAR/s")
    print(f"PERCLOS: {perc:.1f}%")

    risk = 0
    if mean_dur > BLINK_DURATION_THRESH:
        risk += 1
    if perc > PERCLOS_THRESH:
        risk += 1
    if len(closing_speeds) > 0 and np.mean(closing_speeds) < CLOSING_SPEED_THRESH:
        risk += 1  # more negative = faster closing? Actually threshold is negative.
    # For simplicity, we'll just use two criteria
    risk = min(risk, 2)
    risk_level = ["Low", "Moderate", "High"][risk]

    print(f"\nFatigue/neurological risk: {risk_level} ({risk}/2)")

    with open("fatigue_screening_result.txt", "w") as f:
        f.write(f"Recording duration: {t[-1]:.2f} s\n")
        f.write(f"Number of blinks: {len(blinks)}\n")
        f.write(f"Blink frequency: {blink_freq:.2f} blinks/min\n")
        f.write(f"Mean blink duration: {mean_dur*1000:.1f} ms\n")
        f.write(f"PERCLOS: {perc:.1f}%\n")
        f.write(f"Risk score: {risk}/2 - {risk_level}\n")

    print("\nReport saved to 'fatigue_screening_result.txt'.")

if __name__ == "__main__":
    main()