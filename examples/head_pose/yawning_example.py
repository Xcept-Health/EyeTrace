"""
Example: Detect yawns from MAR time series.
"""

import numpy as np
import matplotlib.pyplot as plt
from eyetrace.head_pose.yawning import yawn_detection, yawn_frequency

def main():
    # Simulate MAR signal with two yawns
    fs = 30
    t = np.arange(0, 120, 1/fs)  # 2 minutes

    mar = 0.3 * np.ones_like(t)  # baseline
    # First yawn (30-35s)
    mar[900:1050] = 0.7 + 0.1 * np.sin(np.linspace(0, np.pi, 150))
    # Second yawn (70-75s)
    mar[2100:2250] = 0.65 + 0.1 * np.sin(np.linspace(0, np.pi, 150))

    # Add noise
    mar += 0.02 * np.random.randn(len(t))
    mar = np.clip(mar, 0.2, 0.9)

    # Detect yawns
    yawns = yawn_detection(mar, threshold=0.6, min_duration=1.5, frame_rate=fs)
    freq = yawn_frequency(yawns, total_duration=120)

    print(f"Detected {len(yawns)} yawns in 2 minutes: {yawns}")
    print(f"Yawn frequency: {freq:.2f} yawns/min")

    # Plot
    plt.figure(figsize=(12,4))
    plt.plot(t, mar)
    for start, end in yawns:
        plt.axvspan(t[start], t[end], alpha=0.3, color='red')
    plt.xlabel('Time (s)')
    plt.ylabel('MAR')
    plt.title('MAR with detected yawns')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()