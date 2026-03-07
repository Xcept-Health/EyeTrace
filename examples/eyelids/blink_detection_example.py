"""
Example: Detect blinks from a simulated EAR signal.
"""

import numpy as np
import matplotlib.pyplot as plt
from eyetrace.eyelids.blink_detection import detect_blinks, blink_frequency, mean_closure_duration

def main():
    # Simulate an EAR signal with blinks
    fs = 60  # Hz
    t = np.arange(0, 30, 1/fs)  # 30 seconds
    ear = 0.3 * np.ones_like(t)  # baseline

    # Add blinks (sharp drops)
    for i in range(5):
        center = int(np.random.uniform(1, 29) * fs)
        width = int(0.1 * fs)  # 100 ms
        ear[center-width//2:center+width//2] = 0.1
        # Add some noise
    ear += 0.02 * np.random.randn(len(t))
    ear = np.clip(ear, 0.05, 0.5)

    # Detect blinks
    blinks = detect_blinks(ear, threshold=0.2, min_interval_frames=10)

    print(f"Detected {len(blinks)} blinks:")
    for start, end in blinks:
        print(f"  Blink from {t[start]:.2f}s to {t[end]:.2f}s (duration {(t[end]-t[start])*1000:.1f}ms)")

    # Compute frequency
    freq = blink_frequency(blinks, duration_seconds=30.0)
    print(f"Blink frequency: {freq:.2f} blinks/min")

    # Mean closure duration
    mcd = mean_closure_duration(blinks, frame_rate=fs)
    print(f"Mean closure duration: {mcd*1000:.1f} ms")

    # Plot
    plt.figure(figsize=(12,4))
    plt.plot(t, ear)
    for start, end in blinks:
        plt.axvspan(t[start], t[end], alpha=0.3, color='red')
    plt.axhline(0.2, color='k', linestyle='--', label='Threshold')
    plt.xlabel('Time (s)')
    plt.ylabel('EAR')
    plt.title('EAR signal with detected blinks')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()