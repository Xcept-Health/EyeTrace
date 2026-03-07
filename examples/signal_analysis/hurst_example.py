"""
Example: Estimate Hurst exponent for different types of signals.
"""

import numpy as np
import matplotlib.pyplot as plt
from eyetrace.signal_analysis.hurst import hurst_exponent

def main():
    np.random.seed(42)
    n = 1000

    # 1. White noise (H ~ 0.5)
    white_noise = np.random.randn(n)

    # 2. Persistent (long memory) signal: fractional Brownian motion with H>0.5 (approximated by cumulative sum of noise with persistence)
    # Simple approximation: integrate pink noise? We'll use a rough method
    persistent = np.cumsum(0.5 * np.random.randn(n) + 0.1 * np.sin(np.linspace(0, 20*np.pi, n)))

    # 3. Anti-persistent (mean-reverting): difference of white noise
    anti_persistent = np.diff(white_noise, prepend=0)

    for name, sig in [("White noise", white_noise),
                      ("Persistent (approx)", persistent),
                      ("Anti-persistent", anti_persistent)]:
        H = hurst_exponent(sig)
        print(f"Hurst exponent for {name}: {H:.3f}")

    # Plot the signals
    plt.figure(figsize=(12, 8))
    plt.subplot(3,1,1); plt.plot(white_noise); plt.title('White noise')
    plt.subplot(3,1,2); plt.plot(persistent); plt.title('Persistent')
    plt.subplot(3,1,3); plt.plot(anti_persistent); plt.title('Anti-persistent')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()