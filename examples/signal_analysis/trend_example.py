"""
Example: Estimate linear trend slope of a signal.
"""

import numpy as np
import matplotlib.pyplot as plt
from eyetrace.signal_analysis.trend import trend_slope

def main():
    t = np.linspace(0, 10, 200)
    # Signal with upward trend
    trend = 0.2 * t
    oscillation = np.sin(2 * np.pi * 0.5 * t)
    noise = 0.1 * np.random.randn(len(t))
    signal = trend + oscillation + noise

    slope = trend_slope(signal, times=t)
    print(f"Estimated trend slope: {slope:.4f} (true trend: 0.2)")

    # Remove trend
    estimated_trend = slope * t
    detrended = signal - estimated_trend

    plt.figure(figsize=(10,6))
    plt.subplot(2,1,1)
    plt.plot(t, signal, label='Original')
    plt.plot(t, estimated_trend, 'r--', label=f'Estimated trend (slope={slope:.3f})')
    plt.legend()
    plt.grid(True)
    plt.subplot(2,1,2)
    plt.plot(t, detrended)
    plt.title('Detrended signal')
    plt.xlabel('Time')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()