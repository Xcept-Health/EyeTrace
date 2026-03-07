"""
Example: Saccade detection and analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from eyetrace.gaze.saccades import (
    saccade_velocity,
    saccade_acceleration,
    detect_saccades,
    saccade_fixation_ratio
)

def main():
    # Simulate a gaze signal with saccades
    fs = 100  # Hz
    t = np.arange(0, 5, 1/fs)
    n = len(t)

    # Base gaze (sinusoidal smooth pursuit)
    base_x = 500 + 200 * np.sin(2 * np.pi * 0.5 * t)
    base_y = 400 + 150 * np.cos(2 * np.pi * 0.5 * t)

    # Add saccades at specific times
    gaze_x = base_x.copy()
    gaze_y = base_y.copy()

    # Saccade 1 at t=1.0s
    idx1 = int(1.0 * fs)
    gaze_x[idx1:] += 100
    gaze_y[idx1:] += 50

    # Saccade 2 at t=2.5s
    idx2 = int(2.5 * fs)
    gaze_x[idx2:] -= 80
    gaze_y[idx2:] -= 30

    # Saccade 3 at t=4.0s
    idx3 = int(4.0 * fs)
    gaze_x[idx3:] += 60
    gaze_y[idx3:] -= 40

    # Convert to 3D vectors (just for velocity calculation)
    # We'll use the 2D positions as if they were in a plane; but saccade_velocity expects 3D.
    # We'll create a 3D array with z=0.
    gaze_3d = np.column_stack((gaze_x, gaze_y, np.zeros(n)))

    # Compute velocity
    vel = saccade_velocity(gaze_3d, t, smooth=True)

    # Detect saccades (threshold in rad/s; we need to convert from pixels to radians? Not trivial.
    # For demo, we'll use a simple threshold on the magnitude of the 2D displacement per time.
    # We'll use a different approach: compute 2D velocity in pixels/s and threshold.
    # But since our saccade_velocity gives rad/s (from 3D vectors), we need to adapt.
    # To keep it simple, we'll compute 2D velocity manually.
    dx = np.diff(gaze_x, prepend=gaze_x[0])
    dy = np.diff(gaze_y, prepend=gaze_y[0])
    dt = t[1] - t[0]
    speed_2d = np.sqrt(dx**2 + dy**2) / dt
    is_saccade = speed_2d > 200  # pixels/s threshold

    # Label saccades
    from scipy.ndimage import label
    labeled, n_sacc = label(is_saccade)
    saccades = []
    for i in range(1, n_sacc+1):
        region = np.where(labeled == i)[0]
        if len(region) > 3:  # at least 3 frames
            saccades.append((region[0], region[-1]))

    print(f"Detected {len(saccades)} saccades:")
    for start, end in saccades:
        print(f"  Saccade from {t[start]:.2f}s to {t[end]:.2f}s, duration={(t[end]-t[start])*1000:.1f}ms")

    # Ratio
    ratio = saccade_fixation_ratio(saccades, n)
    print(f"Saccade/fixation ratio: {ratio:.3f}")

    # Plot
    plt.figure(figsize=(12,8))
    plt.subplot(3,1,1)
    plt.plot(t, gaze_x, label='X')
    plt.plot(t, gaze_y, label='Y')
    plt.ylabel('Position (pixels)')
    plt.legend()
    plt.grid(True)

    plt.subplot(3,1,2)
    plt.plot(t, speed_2d)
    plt.axhline(200, color='r', linestyle='--', label='Threshold')
    for start, end in saccades:
        plt.axvspan(t[start], t[end], alpha=0.3, color='red')
    plt.ylabel('Speed (pixels/s)')
    plt.legend()
    plt.grid(True)

    plt.subplot(3,1,3)
    plt.plot(t, vel)  # rad/s from 3D function (may be noisy)
    plt.ylabel('Angular velocity (rad/s)')
    plt.xlabel('Time (s)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()