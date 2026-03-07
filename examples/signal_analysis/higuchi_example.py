"""
Example: Compute Higuchi Fractal Dimension of a time series.
"""

import numpy as np
import matplotlib.pyplot as plt
from eyetrace.signal_analysis.higuchi import higuchi_fractal_dimension

def main():
    # Generate two signals with different complexity
    t = np.linspace(0, 10, 1000)
    smooth = np.sin(2 * np.pi * 1 * t)               # low complexity
    rough = np.cumsum(np.random.randn(len(t)))       # random walk (high complexity)

    hfd_smooth = higuchi_fractal_dimension(smooth)
    hfd_rough = higuchi_fractal_dimension(rough)

    print(f"Higuchi FD of sine wave: {hfd_smooth:.3f}")
    print(f"Higuchi FD of random walk: {hfd_rough:.3f}")

    plt.figure(figsize=(12, 4))
    plt.subplot(1,2,1)
    plt.plot(t, smooth)
    plt.title(f'Sine wave (FD={hfd_smooth:.3f})')
    plt.subplot(1,2,2)
    plt.plot(t, rough)
    plt.title(f'Random walk (FD={hfd_rough:.3f})')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()