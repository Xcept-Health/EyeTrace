"""
Example: Pupillary Light Reflex (PLR) analysis on simulated data.
"""

import numpy as np
import matplotlib.pyplot as plt
from eyetrace.pupil.plr import plr_analysis

def simulate_plr_response(fs=100.0):
    """Simulate a typical PLR response."""
    t = np.arange(0, 8, 1/fs)  # 8 seconds
    stimulus_time = 2.0

    # Baseline
    diameters = 5.0 * np.ones_like(t)

    # Constriction (delay 0.3s, duration 1.0s)
    onset_idx = int((stimulus_time + 0.3) * fs)
    constriction_end = int((stimulus_time + 1.3) * fs)
    if constriction_end < len(t):
        # Smooth constriction
        constriction = np.linspace(5.0, 3.2, constriction_end - onset_idx)
        diameters[onset_idx:constriction_end] = constriction
        # Slow redilation
        redilation = np.linspace(3.2, 4.5, len(t) - constriction_end)
        diameters[constriction_end:] = redilation

    # Add noise
    diameters += 0.05 * np.random.randn(len(t))
    return t, diameters, stimulus_time

def main():
    t, diameters, stim_time = simulate_plr_response()

    # Perform PLR analysis
    results = plr_analysis(
        diameters, t, stim_time,
        baseline_duration=1.0,
        response_window=3.0,
        recovery_level=0.75
    )

    print("=== Pupillary Light Reflex Analysis ===")
    print(f"Baseline mean: {results['baseline_mean']:.2f} mm")
    print(f"Min diameter: {results['min_diameter']:.2f} mm")
    print(f"Amplitude: {results['amplitude']:.2f} mm ({results['amplitude_percent']:.1f}%)")
    print(f"Latency: {results['latency']:.3f} s")
    print(f"Max constriction speed: {results['max_constriction_speed']:.2f} mm/s")
    print(f"Max dilation speed: {results['max_dilation_speed']:.2f} mm/s")
    print(f"75% recovery time: {results['recovery_time_75']:.3f} s")

    # Plot
    plt.figure(figsize=(10,5))
    plt.plot(t, diameters)
    plt.axvline(stim_time, color='r', linestyle='--', label='Stimulus')
    plt.axhline(results['baseline_mean'], color='g', linestyle=':', label='Baseline')
    plt.scatter(t[results['min_index']], results['min_diameter'], color='b', zorder=5, label='Min')
    if results['recovery_index_75'] is not None:
        plt.scatter(t[results['recovery_index_75']],
                    diameters[results['recovery_index_75']],
                    color='orange', label='75% recovery')
    plt.xlabel('Time (s)')
    plt.ylabel('Diameter (mm)')
    plt.title('Pupillary Light Reflex')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()