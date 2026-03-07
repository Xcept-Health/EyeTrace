"""
Example: Estimate neck flexion angle from head pose.
"""

import numpy as np
from eyetrace.head_pose.neck_angle import neck_flexion_angle

# Reuse the same SimulatedLandmarks class as in angles_example
class SimulatedLandmarks:
    def __init__(self, indices, points_norm):
        self.landmark = [self.Landmark(0, 0) for _ in range(468)]
        for idx, (x, y) in zip(indices, points_norm):
            self.landmark[idx] = self.Landmark(x, y)

    class Landmark:
        def __init__(self, x, y, z=0.0):
            self.x = x
            self.y = y
            self.z = z

def generate_simulated_landmarks(img_w, img_h, pitch_deg=0, roll_deg=0, yaw_deg=0):
    """Same generator as in angles_example."""
    indices = [1, 152, 33, 263, 61, 291]
    base_points = np.array([
        [0.5, 0.5], [0.5, 0.85], [0.35, 0.45], [0.65, 0.45], [0.4, 0.7], [0.6, 0.7]
    ])
    rot = np.radians([pitch_deg, roll_deg, yaw_deg])
    dx = rot[2] * 0.1
    dy = rot[0] * 0.1
    points_norm = base_points + np.array([dx, dy])
    points_norm = np.clip(points_norm, 0, 1)
    return SimulatedLandmarks(indices, points_norm)

def main():
    img_w, img_h = 640, 480

    # Simulate head pitched down (flexion)
    landmarks = generate_simulated_landmarks(img_w, img_h, pitch_deg=20)

    # Compute neck flexion angle (should be positive for head down)
    flexion = neck_flexion_angle(landmarks, img_w, img_h)
    print(f"Neck flexion angle: {np.degrees(flexion):.1f}°")

if __name__ == "__main__":
    main()