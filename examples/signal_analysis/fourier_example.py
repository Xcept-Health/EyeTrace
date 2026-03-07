"""
Example: Compute power spectrum, band power, and LF/HF ratio.
"""

import numpy as np
import matplotlib.pyplot as plt
from eyetrace.signal_analysis.fourier import power_spectrum, band_power, lf_hf_ratio

def main():
    fs = 100.0  # Sampling frequency (Hz)
    t = np.arange(0, 10, 1/fs)
    # Signal containing two frequencies: 0.1 Hz and 0.3 Hz
    signal = np.sin(2 * np.pi * 0.1 * t) + 0.5 * np.sin(2 * np.pi * 0.3 * t) + 0.1 * np.random.randn(len(t))

    # Compute power spectrum (Welch's method)
    freqs, psd = power_spectrum(signal, fs, method='welch')

    # Band power
    lf_band = (0.04, 0.15)
    hf_band = (0.15, 0.4)
    lf_power = band_power(freqs, psd, lf_band)
    hf_power = band_power(freqs, psd, hf_band)
    lfhf = lf_power / hf_power if hf_power > 0 else np.inf

    print(f"LF power ({lf_band} Hz): {lf_power:.3f}")
    print(f"HF power ({hf_band} Hz): {hf_power:.3f}")
    print(f"LF/HF ratio: {lfhf:.3f}")

    # Plot spectrum
    plt.figure(figsize=(10, 5))
    plt.semilogy(freqs, psd)
    plt.axvspan(lf_band[0], lf_band[1], alpha=0.2, color='blue', label='LF band')
    plt.axvspan(hf_band[0], hf_band[1], alpha=0.2, color='red', label='HF band')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    plt.title('Power Spectrum')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()