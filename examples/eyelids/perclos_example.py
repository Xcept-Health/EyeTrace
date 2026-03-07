"""
Example: Compute PERCLOS (percentage of eye closure).
"""

import numpy as np
import matplotlib.pyplot as plt
from eyetrace.eyelids.perclos import perclos

def main():
    fs = 30  # Hz
    t = np.arange(0, 180, 1/fs)  # 3 minutes

    # Simulate EAR with drowsiness in the last minute
    ear = 0.3 + 0.02 * np.random.randn(len(t))
    # Last minute: more low EAR
    ear[-60*fs:] = np.maximum(0.15, ear[-60*fs:] - 0.1)
    ear = np.clip(ear, 0.1, 0.4)

    # Compute PERCLOS over a 60-second window
    perc = perclos(ear, threshold=0.2, window_seconds=60, frame_rate=fs)

    # Also compute for the whole sequence
    perc_total = perclos(ear, threshold=0.2, window_seconds=len(t)/fs, frame_rate=fs)

    print(f"PERCLOS (last 60s): {perc:.1f}%")
    print(f"PERCLOS (entire 3min): {perc_total:.1f}%")

    # Plot
    plt.figure(figsize=(12,4))
    plt.plot(t, ear)
    plt.axhline(0.2, color='r', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('EAR')
    plt.title('EAR signal')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()