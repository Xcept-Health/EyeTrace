"""
Example: Extract eye landmarks from simulated MediaPipe face landmarks.
"""

import numpy as np
from eyetrace.eyelids.utils import extract_eye_landmarks_from_mediapipe, extract_both_eyes

# Simulate MediaPipe landmarks (simplified)
class SimulatedLandmarks:
    def __init__(self, indices, points_norm):
        self.landmark = [self.Landmark(0,0) for _ in range(468)]
        for idx, (x, y) in zip(indices, points_norm):
            self.landmark[idx] = self.Landmark(x, y)
    class Landmark:
        def __init__(self, x, y):
            self.x = x
            self.y = y

def generate_face_landmarks(img_w, img_h):
    # Indices for both eyes (simplified)
    left_idx = [33, 160, 158, 133, 153, 144]
    right_idx = [362, 385, 387, 263, 373, 380]
    # Approximate normalized positions for a frontal face
    left_points = np.array([
        [0.3, 0.4], [0.32, 0.38], [0.34, 0.4],
        [0.34, 0.42], [0.32, 0.44], [0.3, 0.42]
    ])
    right_points = np.array([
        [0.7, 0.4], [0.68, 0.38], [0.66, 0.4],
        [0.66, 0.42], [0.68, 0.44], [0.7, 0.42]
    ])
    # Combine all indices and points
    indices = left_idx + right_idx
    points_norm = np.vstack((left_points, right_points))
    return SimulatedLandmarks(indices, points_norm)

def main():
    img_w, img_h = 640, 480
    face_lm = generate_face_landmarks(img_w, img_h)

    # Extract left eye
    left_eye = extract_eye_landmarks_from_mediapipe(face_lm, img_w, img_h, 'left')
    print("Left eye landmarks (pixels):\n", left_eye)

    # Extract both eyes
    left, right = extract_both_eyes(face_lm, img_w, img_h)
    print("\nRight eye landmarks:\n", right)

if __name__ == "__main__":
    main()