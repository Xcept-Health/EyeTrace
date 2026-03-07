"""
Example: Live webcam dashboard with dynamic eye tracking using MediaPipe.
Calculates real EAR and simulates pupil diameter based on eye region.
"""

import cv2
import numpy as np
import time
from eyetrace.io import WebcamReader
from eyetrace.visualization import Dashboard, draw_text_overlay, draw_eye_landmarks

# Attempt to import MediaPipe
try:
    import mediapipe as mp
    MP_AVAILABLE = True
    print("MediaPipe loaded – dynamic tracking enabled")
except ImportError:
    MP_AVAILABLE = False
    print("MediaPipe not installed – will use simulated landmarks")

# Attempt to import real EAR function if available
try:
    from eyetrace.eyelids.ear import ear as ear_func
    EAR_AVAILABLE = True
    print("Real EAR function found")
except ImportError:
    EAR_AVAILABLE = False
    print("Real EAR function not available – computing EAR via simple formula")


def get_eye_landmarks_from_mediapipe(frame, face_mesh):
    """
    Use MediaPipe Face Mesh to extract eye landmarks.
    Returns (left_eye, right_eye) arrays of shape (6,2) or (None, None) if no face.
    """
    if not MP_AVAILABLE or face_mesh is None:
        return None, None

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None, None

    h, w = frame.shape[:2]
    landmarks = results.multi_face_landmarks[0].landmark

    # MediaPipe indices for eye contours (simplified to 6 points each)
    left_idx = [33, 133, 157, 158, 159, 160]
    right_idx = [362, 263, 387, 386, 385, 384]

    left_eye = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in left_idx])
    right_eye = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in right_idx])

    return left_eye.astype(np.float32), right_eye.astype(np.float32)


def compute_ear_from_landmarks(eye_landmarks):
    """
    Compute EAR (Eye Aspect Ratio) from 6 landmarks of one eye.
    Uses real function if available, otherwise a simple formula.
    """
    if EAR_AVAILABLE:
        return ear_func(eye_landmarks)
    else:
        # Simplified EAR formula: vertical distances / horizontal distance
        p1, p2, p3, p4, p5, p6 = eye_landmarks
        vert1 = np.linalg.norm(p2 - p6)
        vert2 = np.linalg.norm(p3 - p5)
        horiz = np.linalg.norm(p1 - p4)
        return (vert1 + vert2) / (2.0 * horiz + 1e-6)


def compute_metrics(left_eye, right_eye):
    """Compute average EAR and simulate pupil diameter."""
    if left_eye is None or right_eye is None:
        return 0.3, 4.0  # default values

    ear_left = compute_ear_from_landmarks(left_eye)
    ear_right = compute_ear_from_landmarks(right_eye)
    ear_value = (ear_left + ear_right) / 2.0

    # Simulate pupil diameter based on time (realistic variation)
    pupil_value = 4.0 + 0.5 * np.sin(time.time() / 3) + 0.1 * np.random.randn()
    pupil_value = np.clip(pupil_value, 3.0, 5.0)

    return ear_value, pupil_value


def main():
    # Initialize MediaPipe if available
    face_mesh = None
    if MP_AVAILABLE:
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,  # to get iris landmarks
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    # Video source (webcam)
    video = WebcamReader(camera_id=0, width=640, height=480)

    # Plot specifications
    plot_specs = [
        {'title': 'Eye Aspect Ratio (EAR)', 'ylabel': 'ratio', 'color': 'b-'},
        {'title': 'Pupil Diameter', 'ylabel': 'mm', 'color': 'g-'}
    ]

    def process_frame(frame):
        """Process each frame: tracking with MediaPipe and metric computation."""
        if MP_AVAILABLE and face_mesh is not None:
            left_eye, right_eye = get_eye_landmarks_from_mediapipe(frame, face_mesh)
        else:
            # Fallback: simulated landmarks (static but with noise)
            left_eye, right_eye = simulate_eye_landmarks(frame)

        ear, pupil = compute_metrics(left_eye, right_eye)

        # Annotate frame
        if left_eye is not None and right_eye is not None:
            annotated = draw_eye_landmarks(frame, left_eye, right_eye,
                                           color_left=(255, 255, 0),
                                           color_right=(0, 255, 255))
        else:
            annotated = frame.copy()
            annotated = draw_text_overlay(annotated, ["No face detected"],
                                          position=(10, 30), color=(0, 0, 255))

        annotated = draw_text_overlay(annotated, [
            f"EAR: {ear:.3f}",
            f"Pupil: {pupil:.2f} mm"
        ], position=(10, 60))

        return {
            'frame': annotated,
            'timestamp': time.time(),
            'metrics': [ear, pupil]
        }

    # Launch dashboard
    with Dashboard(video, plot_specs, update_interval_ms=50,
                   layout='horizontal', window_name='EyeTrace Live - Dynamic') as dashboard:
        print("Dashboard started. Press 'q' in the window to quit.")
        dashboard.run(process_frame)

    # Release MediaPipe resources
    if MP_AVAILABLE and face_mesh is not None:
        face_mesh.close()


def simulate_eye_landmarks(frame):
    """Generate simulated eye landmarks (fallback when MediaPipe is not available or no face detected)."""
    h, w = frame.shape[:2]
    left_center = (w * 0.3, h * 0.4)
    right_center = (w * 0.7, h * 0.4)

    angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
    rx_left, ry_left = 20, 15
    rx_right, ry_right = 20, 15

    left_eye = np.array([
        [left_center[0] + rx_left * np.cos(a), left_center[1] + ry_left * np.sin(a)]
        for a in angles
    ]).astype(np.float32)

    right_eye = np.array([
        [right_center[0] + rx_right * np.cos(a), right_center[1] + ry_right * np.sin(a)]
        for a in angles
    ]).astype(np.float32)

    # Add random movement to simulate dynamics
    left_eye += np.random.randn(*left_eye.shape) * 2
    right_eye += np.random.randn(*right_eye.shape) * 2

    return left_eye, right_eye


if __name__ == "__main__":
    main()