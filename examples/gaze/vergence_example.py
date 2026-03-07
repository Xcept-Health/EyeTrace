"""
Example: Compute vergence angle and speed.
"""

import numpy as np
import matplotlib.pyplot as plt
from eyetrace.gaze.vergence import vergence_angle, vergence_speed

def main():
    # Simulate left and right gaze vectors over time
    fs = 60  # Hz
    t = np.arange(0, 5, 1/fs)
    n = len(t)

    # Both eyes pointing straight ahead, then converge, then diverge
    left_gaze = np.zeros((n, 3))
    right_gaze = np.zeros((n, 3))

    for i, ti in enumerate(t):
        # Left eye: initially straight, then turns right
        left_gaze[i] = [np.sin(0.5 * ti * 0.5), 0, np.cos(0.5 * ti * 0.5)]
        # Right eye: initially straight, then turns left
        right_gaze[i] = [-np.sin(0.5 * ti * 0.5), 0, np.cos(0.5 * ti * 0.5)]

    # Normalize
    left_gaze = left_gaze / np.linalg.norm(left_gaze, axis=1, keepdims=True)
    right_gaze = right_gaze / np.linalg.norm(right_gaze, axis=1, keepdims=True)

    # Compute vergence angle
    angle = vergence_angle(left_gaze, right_gaze)

    # Compute vergence speed
    speed = vergence_speed(left_gaze, right_gaze, t, smooth=True)

    # Plot
    plt.figure(figsize=(10,5))
    plt.subplot(2,1,1)
    plt.plot(t, np.degrees(angle))
    plt.ylabel('Vergence angle (deg)')
    plt.grid(True)

    plt.subplot(2,1,2)
    plt.plot(t, np.degrees(speed))
    plt.ylabel('Vergence speed (deg/s)')
    plt.xlabel('Time (s)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"Mean vergence angle: {np.mean(angle):.3f} rad")

if __name__ == "__main__":
    main()