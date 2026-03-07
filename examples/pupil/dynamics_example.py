"""
Example: Compute pupil dynamics: velocity, constriction/dilation speeds, hippus amplitude.
"""

import numpy as np
import matplotlib.pyplot as plt
from eyetrace.pupil.dynamics import (
    first_derivative,
    constriction_speed,
    dilation_speed,
    hippus_amplitude
)

def main():
    # Simulate a pupil signal with a constriction event and oscillations
    fs = 50.0  # Hz
    t = np.arange(0, 10, 1/fs)

    # Baseline with slow oscillation (hippus)
    base = 4.0 + 0.2 * np.sin(2 * np.pi * 1.2 * t)  # 1.2 Hz hippus

    # Add a constriction-dilation event
    event_start = 3.0  # seconds
    event_duration = 2.0
    constriction_amp = -1.5  # mm
    event = constriction_amp * np.exp(-((t - event_start) / 0.3)**2)  # Gaussian dip
    event += 0.8 * constriction_amp * np.exp(-((t - (event_start+0.8)) / 0.5)**2)  # secondary

    diameters = base + event
    # Add noise
    diameters += 0.05 * np.random.randn(len(t))

    # First derivative (velocity)
    vel = first_derivative(diameters, t, smooth=True)

    # Constriction speed
    max_const_speed, avg_const_speed = constriction_speed(diameters, t, threshold=1.0)
    max_dil_speed, avg_dil_speed = dilation_speed(diameters, t, threshold=1.0)

    print(f"Max constriction speed: {max_const_speed:.2f} mm/s")
    print(f"Avg constriction speed: {avg_const_speed:.2f} mm/s")
    print(f"Max dilation speed: {max_dil_speed:.2f} mm/s")
    print(f"Avg dilation speed: {avg_dil_speed:.2f} mm/s")

    # Hippus amplitude
    hippus = hippus_amplitude(diameters, fs, lowcut=0.5, highcut=4.0, method='rms')
    print(f"Hippus amplitude (RMS): {hippus:.3f} mm")

    # Plot
    plt.figure(figsize=(12, 8))
    plt.subplot(2,1,1)
    plt.plot(t, diameters)
    plt.title('Pupil Diameter')
    plt.ylabel('Diameter (mm)')
    plt.grid(True)

    plt.subplot(2,1,2)
    plt.plot(t, vel)  # gradient returns same length
    plt.axhline(0, color='k', linestyle='--')
    plt.title('Velocity')
    plt.xlabel('Time (s)')
    plt.ylabel('mm/s')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()