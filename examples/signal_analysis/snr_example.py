"""
Example: Estimate Signal-to-Noise Ratio using different methods.
"""

import numpy as np
import matplotlib.pyplot as plt
from eyetrace.signal_analysis.snr import signal_to_noise_ratio

def main():
    # Generate a clean signal and add noise
    t = np.linspace(0, 10, 500)
    clean = np.sin(2 * np.pi * 2 * t)
    noise = 0.3 * np.random.randn(len(t))
    noisy = clean + noise

    # SNR estimation
    snr_std = signal_to_noise_ratio(noisy, method='standard')
    snr_smooth = signal_to_noise_ratio(noisy, method='smooth')

    print(f"SNR (standard mean/std): {snr_std:.3f}")
    print(f"SNR (smooth-based): {snr_smooth:.3f}")

    # Theoretical SNR (based on known noise std)
    theoretical = np.std(clean) / np.std(noise)
    print(f"Theoretical SNR: {theoretical:.3f}")

    plt.figure(figsize=(10,4))
    plt.plot(t, clean, label='Clean', alpha=0.7)
    plt.plot(t, noisy, label='Noisy', alpha=0.7)
    plt.legend()
    plt.title(f'Signal (SNR≈{snr_smooth:.2f})')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()