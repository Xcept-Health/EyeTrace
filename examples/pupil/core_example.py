"""
Example: Extract pupil diameter and iris radius from simulated iris landmarks.
"""

import numpy as np
from eyetrace.pupil.core import extract_pupil_diameter, extract_iris_radius

def simulate_iris_landmarks(center_x, center_y, radius, num_points=5):
    """Generate 5 points around an iris (simulating MediaPipe output)."""
    angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    landmarks = np.array([
        [center_x + radius * np.cos(a), center_y + radius * np.sin(a)]
        for a in angles
    ])
    return landmarks.astype(np.float32)

def main():
    # Simulate image size and iris landmarks
    img_w, img_h = 640, 480
    # Left iris
    left_iris = simulate_iris_landmarks(300, 240, radius=30)
    # Right iris
    right_iris = simulate_iris_landmarks(340, 240, radius=30)

    # Extract pupil diameter (assuming pupil coincides with iris center)
    left_dia = extract_pupil_diameter(left_iris, img_w, img_h, px_to_mm=0.05)  # 0.05 mm per pixel
    right_dia = extract_pupil_diameter(right_iris, img_w, img_h, px_to_mm=0.05)

    # Extract iris radius
    left_rad = extract_iris_radius(left_iris, img_w, img_h, px_to_mm=0.05)
    right_rad = extract_iris_radius(right_iris, img_w, img_h, px_to_mm=0.05)

    print(f"Left pupil diameter: {left_dia:.2f} mm")
    print(f"Right pupil diameter: {right_dia:.2f} mm")
    print(f"Left iris radius: {left_rad:.2f} mm")
    print(f"Right iris radius: {right_rad:.2f} mm")

if __name__ == "__main__":
    main()