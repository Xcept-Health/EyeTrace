"""
Example: Compute EAR jerk (second derivative).
"""

import numpy as np
import matplotlib.pyplot as plt
from eyetrace.eyelids.jerk import ear_jerk
from eyetrace.eyelids.blink_detection import detect_blinks

def main():
    fs = 100  # Hz
    t = np.arange(0, 5, 1/fs)

    # Create EAR signal with a blink
    ear = 0.3 * np.ones_like(t)
    blink_center = int(2.0 * fs)
    blink_duration = int(0.2 * fs)
    ear[blink_center-blink_duration//2 : blink_center+blink_duration//2] = 0.1
    ear[blink_center-blink_duration//2 : blink_center] = np.linspace(0.3, 0.1, blink_duration//2)
    ear[blink_center : blink_center+blink_duration//2] = np.linspace(0.1, 0.3, blink_duration//2)

    # Compute jerk
    jerk = ear_jerk(ear, t, smooth=True)

    # Plot
    plt.figure(figsize=(12,6))
    plt.subplot(2,1,1)
    plt.plot(t, ear)
    plt.ylabel('EAR')
    plt.grid(True)
    plt.subplot(2,1,2)
    plt.plot(t, jerk)
    plt.xlabel('Time (s)')
    plt.ylabel('Jerk')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()