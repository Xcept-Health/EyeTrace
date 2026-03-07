"""
Example: Compute Eye Aspect Ratio (EAR) from simulated eye landmarks.
"""

import numpy as np
from eyetrace.eyelids.ear import eye_aspect_ratio, both_eyes_ear

def generate_eye_landmarks(center_x, center_y, width=30, height=15, open_factor=1.0):
    """
    Generate 6 eye landmarks around an ellipse.
    Order: p1 (outer corner), p2 (top outer), p3 (top inner), p4 (inner corner),
           p5 (bottom inner), p6 (bottom outer)
    """
    # Ellipse parameters
    a = width / 2   # semi-major axis
    b = height / 2 * open_factor  # semi-minor axis (vertical) controlled by open_factor

    angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
    # Rotate so that p1 is at angle 0 (rightmost point)
    # We want p1 at angle 0, p4 at pi (leftmost)
    angles = angles - angles[0]
    x = center_x + a * np.cos(angles)
    y = center_y + b * np.sin(angles)
    return np.column_stack((x, y))

def main():
    # Simulate left eye (open)
    left_eye = generate_eye_landmarks(300, 240, width=40, height=20, open_factor=1.0)
    # Simulate right eye (open)
    right_eye = generate_eye_landmarks(340, 240, width=40, height=20, open_factor=1.0)

    ear_left = eye_aspect_ratio(left_eye)
    ear_right = eye_aspect_ratio(right_eye)
    ear_avg = both_eyes_ear(left_eye, right_eye)

    print(f"Left eye EAR: {ear_left:.3f}")
    print(f"Right eye EAR: {ear_right:.3f}")
    print(f"Average EAR: {ear_avg:.3f}")

    # Simulate half-closed eyes
    left_eye_half = generate_eye_landmarks(300, 240, width=40, height=20, open_factor=0.5)
    right_eye_half = generate_eye_landmarks(340, 240, width=40, height=20, open_factor=0.5)
    ear_avg_half = both_eyes_ear(left_eye_half, right_eye_half)
    print(f"Average EAR (half closed): {ear_avg_half:.3f}")

if __name__ == "__main__":
    main()