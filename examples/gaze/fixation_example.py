"""
Example: Fixation analysis: duration, dispersion, centroid.
"""

import numpy as np
from scipy.ndimage import label
import matplotlib.pyplot as plt
from eyetrace.gaze.fixation import fixation_duration, fixation_dispersion, gaze_centroid

def main():
    # Simulate gaze data with fixations (simplified)
    fs = 60  # Hz
    t = np.arange(0, 10, 1/fs)
    n = len(t)

    # Create a boolean fixation mask: three fixations
    fixation_mask = np.zeros(n, dtype=bool)
    fixation_mask[100:200] = True   # fixation 1
    fixation_mask[300:450] = True   # fixation 2
    fixation_mask[500:600] = True   # fixation 3

    # Simulate gaze positions with slight jitter during fixations
    gaze_x = np.random.normal(500, 10, n)
    gaze_y = np.random.normal(400, 10, n)
    # Add saccades between fixations (large jumps)
    gaze_x[200:300] += 300
    gaze_y[200:300] += 200
    gaze_x[450:500] -= 200
    gaze_y[450:500] -= 100

    gaze_positions = np.column_stack((gaze_x, gaze_y))

    # Compute fixation metrics
    durations = fixation_duration(fixation_mask, t)
    dispersions = fixation_dispersion(gaze_positions, fixation_mask)
    centroid = gaze_centroid(gaze_positions, fixation_mask)

    print("Fixation analysis:")
    for i, (dur, disp) in enumerate(zip(durations, dispersions)):
        print(f"  Fixation {i+1}: duration={dur:.2f}s, dispersion={disp:.2f} pixels")

    print(f"Overall gaze centroid (fixations only): {centroid}")

    # Plot
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(t, gaze_x, label='X')
    plt.plot(t, gaze_y, label='Y')
    plt.fill_between(t, 0, 1, where=fixation_mask, color='yellow', alpha=0.3, transform=plt.gca().get_xaxis_transform(), label='Fixation')
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Position (pixels)')

    plt.subplot(1,2,2)
    plt.scatter(gaze_x, gaze_y, s=1, alpha=0.5)
    plt.scatter(centroid[0], centroid[1], color='red', s=100, marker='x', label='Centroid')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Gaze positions')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()