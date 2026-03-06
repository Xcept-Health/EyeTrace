#!/usr/bin/env python
"""
Dépistage précoce de la maladie de Parkinson par analyse des micro-mouvements oculaires.
Avec visualisation en temps réel des courbes (OpenCV uniquement).
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    from eyetrace.pupil import extract_pupil_diameter, hippus_amplitude
    from eyetrace.gaze import detect_saccades
except ImportError:
    print("Erreur : EyeTrace doit être installé. Exécutez 'pip install -e .' à la racine du projet.")
    sys.exit(1)

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
DURATION = 30
FS = 30
SACCADE_FREQ_THRESH = 2.0
HIPPUS_THRESH = 0.5
PUPIL_VAR_THRESH = 0.8

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
GRAPH_HEIGHT = 150

# Initialisation MediaPipe
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
    """Dessine une courbe simple avec OpenCV."""
    if len(data) < 2:
        cv2.putText(img, f"{label}: pas de données", (10, y_offset + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return
    arr = np.array(data)
    min_val = np.min(arr)
    max_val = np.max(arr)
    if max_val - min_val == 0:
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
        print("Erreur : webcam non accessible.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT - GRAPH_HEIGHT)

    print("=== Dépistage Parkinson par oculométrie ===")
    print("Veuillez fixer le point blanc à l'écran pendant 30 secondes.")
    print("Appuyez sur ESPACE pour commencer, ou 'q' pour quitter.")

    # Buffers pour l'affichage temps réel
    buf_diam = deque(maxlen=300)
    buf_gaze_x = deque(maxlen=300)
    buf_gaze_y = deque(maxlen=300)

    # Données pour l'analyse finale
    timestamps = []
    diameters = []
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

        # Créer l'image de visualisation
        display = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
        # Fond sombre pour les graphiques
        display[h_frame:, :] = (40, 40, 40)
        display[:h_frame, :w_frame] = frame

        # Point de fixation
        cv2.circle(display, (w_frame//2, h_frame//2), 5, (255, 255, 255), -1)

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
        gaze_center = None

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            left_iris = get_iris_landmarks(face_landmarks, LEFT_IRIS, w_frame, h_frame)
            right_iris = get_iris_landmarks(face_landmarks, RIGHT_IRIS, w_frame, h_frame)

            d_left = extract_pupil_diameter(left_iris, w_frame, h_frame)
            d_right = extract_pupil_diameter(right_iris, w_frame, h_frame)
            diam = (d_left + d_right) / 2.0

            left_center = np.mean(left_iris, axis=0)
            right_center = np.mean(right_iris, axis=0)
            gaze_center = (left_center + right_center) / 2.0

            # Dessiner les iris
            for pt in left_iris:
                cv2.circle(display, tuple(pt.astype(int)), 2, (255, 255, 0), -1)
            for pt in right_iris:
                cv2.circle(display, tuple(pt.astype(int)), 2, (0, 255, 255), -1)

            if recording:
                timestamps.append(current_time)
                diameters.append(diam)
                gaze_x.append(gaze_center[0])
                gaze_y.append(gaze_center[1])

            # Mettre à jour les buffers (même hors enregistrement)
            if diam is not None:
                buf_diam.append(diam)
            if gaze_center is not None:
                buf_gaze_x.append(gaze_center[0])
                buf_gaze_y.append(gaze_center[1])

        # Interface
        if not recording:
            cv2.putText(display, "Appuyez sur ESPACE pour commencer", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            elapsed = time.time() - start_time
            remaining = max(0, DURATION - elapsed)
            cv2.putText(display, f"Enregistrement... {remaining:.1f}s", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if elapsed >= DURATION:
                break

        cv2.putText(display, f"FPS: {fps:.1f}", (WINDOW_WIDTH-100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # Zone des graphiques
        graph_y_start = h_frame + 10
        graph_height = 100
        if buf_diam:
            draw_graph(display, list(buf_diam), (0,255,0), graph_y_start, graph_height, "Diam (px)")
        if buf_gaze_x:
            draw_graph(display, list(buf_gaze_x), (255,100,100), graph_y_start + graph_height + 10, graph_height, "Gaze X (px)")
        if buf_gaze_y:
            draw_graph(display, list(buf_gaze_y), (100,255,255), graph_y_start + 2*(graph_height+10), graph_height, "Gaze Y (px)")

        cv2.imshow("Parkinson Eye Screener - Live Curves", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' ') and not recording:
            recording = True
            start_time = time.time()
            timestamps.clear()
            diameters.clear()
            gaze_x.clear()
            gaze_y.clear()
            # Les buffers sont conservés pour l'affichage
            print("Enregistrement démarré...")

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()

    if len(timestamps) < 10:
        print("Pas assez de données.")
        return

    # Analyse
    t = np.array(timestamps) - timestamps[0]
    d = np.array(diameters)
    gx = np.array(gaze_x)
    gy = np.array(gaze_y)

    print("\n--- Analyse des biomarqueurs ---")
    pupil_cv = np.std(d) / np.mean(d) * 100
    print(f"Coefficient de variation pupillaire : {pupil_cv:.2f}%")
    pupil_var_score = 1 if pupil_cv > PUPIL_VAR_THRESH * 100 else 0

    hippus = hippus_amplitude(d, fs=FS, method='rms')
    print(f"Amplitude hippus : {hippus:.3f} px")
    hippus_score = 1 if hippus > HIPPUS_THRESH else 0

    if len(gx) > 10:
        gaze_pos = np.zeros((len(gx), 3))
        gaze_pos[:, 0] = gx
        gaze_pos[:, 1] = gy
        saccades = detect_saccades(gaze_pos, t, velocity_threshold=1.0)
        saccade_count = len(saccades)
        saccade_freq = saccade_count / t[-1]
        print(f"Micro-saccades détectées : {saccade_count} (fréquence {saccade_freq:.2f} Hz)")
        saccade_score = 1 if saccade_freq > SACCADE_FREQ_THRESH else 0
    else:
        saccade_score = 0
        saccade_freq = 0

    total_score = pupil_var_score + hippus_score + saccade_score
    risk_level = ["Faible", "Modéré", "Élevé", "Très élevé"][total_score]

    print("\n--- RÉSULTATS ---")
    print(f"Score de risque Parkinson : {total_score}/3")
    print(f"Niveau de risque : {risk_level}")

    with open("parkinson_screening_results.txt", "w") as f:
        f.write(f"Durée : {t[-1]:.2f} s\n")
        f.write(f"Coefficient de variation : {pupil_cv:.2f}%\n")
        f.write(f"Amplitude hippus : {hippus:.3f} px\n")
        f.write(f"Fréquence micro-saccades : {saccade_freq:.2f} Hz\n")
        f.write(f"Score total : {total_score}/3\n")
        f.write(f"Niveau de risque : {risk_level}\n")

    print("\nRapport enregistré dans 'parkinson_screening_results.txt'.")

if __name__ == "__main__":
    main()