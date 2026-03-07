"""
Example: Compute Lempel-Ziv complexity of a signal.
"""

import numpy as np
import matplotlib.pyplot as plt
from eyetrace.signal_analysis.lempel_ziv import lempel_ziv_complexity

def main():
    # Generate signals with different complexity
    t = np.linspace(0, 10, 1000)
    simple = np.sin(2 * np.pi * 0.5 * t)               # simple oscillation
    complex_sig = np.cumsum(np.random.randn(len(t)))    # random walk

    # Compute LZ complexity (normalized)
    lz_simple = lempel_ziv_complexity(simple, normalize=True)
    lz_complex = lempel_ziv_complexity(complex_sig, normalize=True)

    print(f"LZ complexity of sine wave: {lz_simple:.4f}")
    print(f"LZ complexity of random walk: {lz_complex:.4f}")

    # Show binarized versions
    simple_bin = (simple > np.median(simple)).astype(int)
    complex_bin = (complex_sig > np.median(complex_sig)).astype(int)

    plt.figure(figsize=(12, 6))
    plt.subplot(2,2,1); plt.plot(t, simple); plt.title('Original sine')
    plt.subplot(2,2,2); plt.plot(t, simple_bin, drawstyle='steps'); plt.title('Binarized sine')
    plt.subplot(2,2,3); plt.plot(t, complex_sig); plt.title('Original random walk')
    plt.subplot(2,2,4); plt.plot(t, complex_bin, drawstyle='steps'); plt.title('Binarized random walk')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()