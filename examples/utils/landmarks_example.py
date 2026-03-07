"""
Example: Extract eye landmarks from an image using MediaPipe.
If MediaPipe not installed, uses simulated landmarks.
"""

import cv2
import numpy as np
from eyetrace.utils.landmarks import extract_eye_landmarks_from_mediapipe, extract_iris_landmarks_from_mediapipe

# Try to import MediaPipe
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False
    print("MediaPipe not installed – using simulated landmarks.")

def simulate_eye_landmarks(image_shape):
    """Generate fake landmarks for demonstration."""
    h, w = image_shape[:2]
    left_eye = np.array([
        [int(w*0.3), int(h*0.4)],
        [int(w*0.35), int(h*0.38)],
        [int(w*0.4), int(h*0.4)],
        [int(w*0.4), int(h*0.45)],
        [int(w*0.35), int(h*0.47)],
        [int(w*0.3), int(h*0.45)]
    ], dtype=np.float64)
    right_eye = np.array([
        [int(w*0.6), int(h*0.4)],
        [int(w*0.65), int(h*0.38)],
        [int(w*0.7), int(h*0.4)],
        [int(w*0.7), int(h*0.45)],
        [int(w*0.65), int(h*0.47)],
        [int(w*0.6), int(h*0.45)]
    ], dtype=np.float64)
    return left_eye, right_eye

def main():
    # Create a blank image or load a real one
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (100, 100, 100)  # gray

    if MP_AVAILABLE:
        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                h, w = img.shape[:2]
                left_eye = extract_eye_landmarks_from_mediapipe(face_landmarks, w, h, 'left')
                right_eye = extract_eye_landmarks_from_mediapipe(face_landmarks, w, h, 'right')
                left_iris = extract_iris_landmarks_from_mediapipe(face_landmarks, w, h, 'left')
                right_iris = extract_iris_landmarks_from_mediapipe(face_landmarks, w, h, 'right')
                print("Real landmarks extracted from MediaPipe.")
            else:
                print("No face detected – using simulated landmarks.")
                left_eye, right_eye = simulate_eye_landmarks(img.shape)
                left_iris, right_iris = None, None
    else:
        left_eye, right_eye = simulate_eye_landmarks(img.shape)
        left_iris = right_iris = None

    # Draw landmarks on image for visualization
    for (x, y) in left_eye.astype(int):
        cv2.circle(img, (x, y), 2, (255, 255, 0), -1)
    for (x, y) in right_eye.astype(int):
        cv2.circle(img, (x, y), 2, (0, 255, 255), -1)
    if left_iris is not None:
        for (x, y) in left_iris.astype(int):
            cv2.circle(img, (x, y), 1, (255, 0, 0), -1)
    if right_iris is not None:
        for (x, y) in right_iris.astype(int):
            cv2.circle(img, (x, y), 1, (0, 0, 255), -1)

    cv2.imshow("Landmarks", img)
    print("Press any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()