"""
Example: Demonstrate filtering functions on a noisy signal.
"""

import numpy as np
import matplotlib.pyplot as plt
from eyetrace.utils.filtering import moving_average, savgol_filter, kalman_filter_1d

def main():
    # Generate a noisy sine wave
    t = np.linspace(0, 10, 200)
    clean = np.sin(t)
    noise = 0.2 * np.random.randn(len(t))
    noisy = clean + noise

    # Apply filters
    ma = moving_average(noisy, window_size=11)
    sg = savgol_filter(noisy, window_length=11, polyorder=2)
    kalman = kalman_filter_1d(noisy, Q=1e-3, R=1e-1)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(t, noisy, 'gray', alpha=0.5, label='Noisy')
    plt.plot(t, clean, 'k--', label='True')
    plt.plot(t, ma, label='Moving Average (11)')
    plt.plot(t, sg, label='Savitzky-Golay (11,2)')
    plt.plot(t, kalman, label='Kalman (Q=1e-3, R=1e-1)')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Filtering Example')
    plt.grid(True)
    plt.show()

    # Print some stats
    print("Original SNR: {:.2f}".format(10 * np.log10(np.var(clean) / np.var(noise))))
    print("MA SNR: {:.2f}".format(10 * np.log10(np.var(clean) / np.var(clean - ma))))
    print("SG SNR: {:.2f}".format(10 * np.log10(np.var(clean) / np.var(clean - sg))))
    print("Kalman SNR: {:.2f}".format(10 * np.log10(np.var(clean) / np.var(clean - kalman))))

if __name__ == "__main__":
    main()