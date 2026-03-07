"""
Example: Detect microsleep events (prolonged eye closure).
"""

import numpy as np
import matplotlib.pyplot as plt
from eyetrace.eyelids.microsleep import microsleep_indicator

def main():
    fs = 30  # Hz
    t = np.arange(0, 120, 1/fs)  # 2 minutes

    # Simulate EAR with normal blinks and a microsleep event
    ear = 0.3 + 0.02 * np.random.randn(len(t))
    # Add a microsleep from 60 to 63 seconds
    micro_start = int(60 * fs)
    micro_end = int(63 * fs)
    ear[micro_start:micro_end] = 0.1
    # Add normal blinks
    for i in range(20):
        pos = np.random.randint(0, len(t))
        ear[pos:pos+5] = 0.15

    ear = np.clip(ear, 0.05, 0.4)

    # Detect microsleeps (≥2 seconds closure)
    micro_mask = microsleep_indicator(ear, frame_rate=fs, ear_threshold=0.2, duration_threshold=2.0)

    print(f"Microsleep frames: {np.sum(micro_mask)} / {len(micro_mask)}")

    # Plot
    plt.figure(figsize=(12,4))
    plt.plot(t, ear)
    plt.fill_between(t, 0, 1, where=micro_mask, color='red', alpha=0.5, transform=plt.gca().get_xaxis_transform())
    plt.axhline(0.2, color='k', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('EAR')
    plt.title('Microsleep detection')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()