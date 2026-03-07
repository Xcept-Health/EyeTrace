"""
Example: Estimate head pose angles (pitch, roll, yaw) from simulated face landmarks.
"""

import numpy as np
import cv2
from eyetrace.head_pose.angles import head_pose_angles

class SimulatedLandmarks:
    """Simulate MediaPipe face landmarks with a full list of 468 points."""
    def __init__(self, indices, points_norm):
        # Create a list of 468 dummy landmarks (all at (0,0) initially)
        self.landmark = [self.Landmark(0, 0) for _ in range(468)]
        # Fill the specified indices
        for idx, (x, y) in zip(indices, points_norm):
            self.landmark[idx] = self.Landmark(x, y)

    class Landmark:
        def __init__(self, x, y, z=0.0):
            self.x = x
            self.y = y
            self.z = z

def generate_simulated_landmarks(img_w, img_h, pitch_deg=0, roll_deg=0, yaw_deg=0):
    """
    Generate normalized landmarks for a generic face with given rotations.
    Returns a SimulatedLandmarks object.
    """
    # Indices: nose tip, chin, left eye left, right eye right, left mouth, right mouth
    indices = [1, 152, 33, 263, 61, 291]
    # Approximate 2D positions (normalized) of a frontal face
    base_points = np.array([
        [0.5, 0.5],      # nose tip
        [0.5, 0.85],     # chin
        [0.35, 0.45],    # left eye left
        [0.65, 0.45],    # right eye right
        [0.4, 0.7],      # left mouth
        [0.6, 0.7]       # right mouth
    ])
    # Apply simple 2D shifts to simulate rotation (very crude)
    rot = np.radians([pitch_deg, roll_deg, yaw_deg])
    dx = rot[2] * 0.1  # yaw -> horizontal shift
    dy = rot[0] * 0.1  # pitch -> vertical shift
    points_norm = base_points + np.array([dx, dy])
    points_norm = np.clip(points_norm, 0, 1)
    return SimulatedLandmarks(indices, points_norm)

def main():
    img_w, img_h = 640, 480

    # Simulate a face looking slightly up and right
    landmarks = generate_simulated_landmarks(img_w, img_h, pitch_deg=-10, yaw_deg=15)

    # Estimate head pose angles
    pitch, roll, yaw = head_pose_angles(landmarks, img_w, img_h)

    print(f"Pitch: {np.degrees(pitch):.1f}°")
    print(f"Roll: {np.degrees(roll):.1f}°")
    print(f"Yaw: {np.degrees(yaw):.1f}°")

if __name__ == "__main__":
    main()