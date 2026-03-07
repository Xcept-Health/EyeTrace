"""
Example: Compute postural sag (eye height trend).
"""

import numpy as np
import matplotlib.pyplot as plt
from eyetrace.head_pose.postural_sag import postural_sag

def main():
    # Simulate eye y-coordinate over 60 seconds
    fs = 10  # Hz
    t = np.arange(0, 60, 1/fs)

    # Start at 300, gradually sag down by 20 pixels over 60s
    y_eye = 300 - 0.33 * t + 5 * np.sin(2 * np.pi * 0.02 * t) + np.random.randn(len(t)) * 2
    y_eye = np.clip(y_eye, 250, 310)

    # Compute sag slope (pixels per second)
    slope = postural_sag(y_eye, t, baseline_seconds=10)
    print(f"Postural sag slope: {slope:.2f} pixels/s (positive = downward)")

    # Plot
    plt.figure(figsize=(10,4))
    plt.plot(t, y_eye)
    plt.xlabel('Time (s)')
    plt.ylabel('Eye Y-coordinate (pixels)')
    plt.title('Eye Height Over Time')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()