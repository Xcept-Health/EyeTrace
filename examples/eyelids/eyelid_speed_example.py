"""
Example: Compute eyelid closing and opening speeds.
"""

import numpy as np
import matplotlib.pyplot as plt
from eyetrace.eyelids.eyelid_speed import eyelid_closing_speed, eyelid_opening_speed
from eyetrace.eyelids.blink_detection import detect_blinks

def main():
    fs = 100  # Hz
    t = np.arange(0, 10, 1/fs)

    # Create EAR signal with a blink
    ear = 0.3 * np.ones_like(t)
    blink_center = int(3.0 * fs)
    blink_duration = int(0.2 * fs)  # 200 ms
    ear[blink_center-blink_duration//2 : blink_center+blink_duration//2] = 0.1
    ear[blink_center-blink_duration//2 : blink_center] = np.linspace(0.3, 0.1, blink_duration//2)
    ear[blink_center : blink_center+blink_duration//2] = np.linspace(0.1, 0.3, blink_duration//2)
    ear += 0.01 * np.random.randn(len(t))
    ear = np.clip(ear, 0.05, 0.35)

    # Detect blinks
    blinks = detect_blinks(ear, threshold=0.2)
    print(f"Blinks: {blinks}")

    # Compute speeds
    closing = eyelid_closing_speed(ear, t, threshold=0.2, smooth=True)
    opening = eyelid_opening_speed(ear, t, threshold=0.2, smooth=True)

    print(f"Closing speeds: {closing}")  # negative values
    print(f"Opening speeds: {opening}")  # positive

    # Plot
    plt.figure(figsize=(12,4))
    plt.plot(t, ear)
    for start, end in blinks:
        plt.axvspan(t[start], t[end], alpha=0.3, color='red')
    plt.xlabel('Time (s)')
    plt.ylabel('EAR')
    plt.title('EAR with blink')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()