"""
Example: Compute Sample Entropy and Approximate Entropy on a synthetic signal.
"""

import numpy as np
import matplotlib.pyplot as plt
from eyetrace.signal_analysis.entropy import sample_entropy, approximate_entropy

def main():
    # Generate a signal: sine wave + noise
    t = np.linspace(0, 10, 500)
    clean = np.sin(2 * np.pi * 0.5 * t)
    noise = 0.2 * np.random.randn(len(t))
    signal = clean + noise

    # Compute entropies
    sampen = sample_entropy(signal, m=2, r=0.2 * np.std(signal))
    apen = approximate_entropy(signal, m=2, r=0.2 * np.std(signal))

    print(f"Sample Entropy (m=2): {sampen:.4f}")
    print(f"Approximate Entropy (m=2): {apen:.4f}")

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(t, signal, label='Signal')
    plt.title(f'Signal (SampEn={sampen:.3f}, ApEn={apen:.3f})')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()