"""
Example: Compute Mouth Aspect Ratio (MAR) from face landmarks.
"""

import numpy as np
from eyetrace.head_pose.mar import mouth_aspect_ratio

class SimulatedLandmarks:
    """Simulate MediaPipe face landmarks with a full list of 468 points."""
    def __init__(self, indices, points_norm):
        self.landmark = [self.Landmark(0, 0) for _ in range(468)]
        for idx, (x, y) in zip(indices, points_norm):
            self.landmark[idx] = self.Landmark(x, y)

    class Landmark:
        def __init__(self, x, y):
            self.x = x
            self.y = y

def generate_mouth_landmarks(img_w, img_h, mouth_open=0.3):
    """
    Generate normalized landmarks for mouth: top(13), bottom(14), left(61), right(291).
    """
    # Base positions for closed mouth
    base = {
        13: [0.5, 0.55],  # top inner lip
        14: [0.5, 0.6],   # bottom inner lip
        61: [0.4, 0.575], # left corner
        291: [0.6, 0.575] # right corner
    }
    # Open mouth: increase vertical distance
    openness = mouth_open * 0.05
    points = base.copy()
    points[13][1] -= openness
    points[14][1] += openness

    indices = [13, 14, 61, 291]
    points_norm = [points[i] for i in indices]
    return SimulatedLandmarks(indices, points_norm)

def main():
    img_w, img_h = 640, 480

    # Closed mouth
    landmarks_closed = generate_mouth_landmarks(img_w, img_h, mouth_open=0.0)
    mar_closed = mouth_aspect_ratio(landmarks_closed, img_w, img_h)
    print(f"MAR (closed mouth): {mar_closed:.3f}")

    # Open mouth
    landmarks_open = generate_mouth_landmarks(img_w, img_h, mouth_open=1.0)
    mar_open = mouth_aspect_ratio(landmarks_open, img_w, img_h)
    print(f"MAR (open mouth): {mar_open:.3f}")

if __name__ == "__main__":
    main()