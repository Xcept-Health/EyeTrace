"""
Example: Compute gaze entropy from simulated scanpath.
"""

import numpy as np
import matplotlib.pyplot as plt
from eyetrace.gaze.entropy import gaze_entropy

def main():
    # Simulate gaze positions on a screen (e.g., 1920x1080)
    np.random.seed(42)
    n = 1000
    x = np.random.normal(960, 300, n)
    y = np.random.normal(540, 200, n)
    gaze_positions = np.column_stack((x, y))

    # Compute entropy with 20x20 bins
    entropy = gaze_entropy(gaze_positions, bins=20)
    print(f"Gaze entropy (20x20 bins): {entropy:.3f} bits")

    # Plot histogram
    plt.figure(figsize=(8,6))
    plt.hist2d(x, y, bins=20, cmap='hot')
    plt.colorbar(label='Count')
    plt.title(f'Gaze Heatmap (Entropy = {entropy:.2f} bits)')
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    plt.show()

if __name__ == "__main__":
    main()