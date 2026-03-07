"""
Example: Demonstrate mathematical helpers: outlier removal, sliding window, peak detection.
"""

import numpy as np
from eyetrace.utils.math_helpers import mad_outlier_removal, sliding_window_view, find_peaks

def main():
    # Generate data with outliers
    np.random.seed(42)
    data = np.sin(np.linspace(0, 4*np.pi, 100)) + 0.1 * np.random.randn(100)
    # Add outliers
    data[20] += 5
    data[50] -= 4
    data[80] += 3

    # Outlier removal
    mask, cleaned = mad_outlier_removal(data, threshold=3.5)
    print("Outlier removal (MAD):")
    print(f"  Original data shape: {data.shape}")
    print(f"  Number of outliers: {np.sum(~mask)}")
    print(f"  Cleaned data (with NaNs): {cleaned[:5]} ...")

    # Sliding window
    window_size = 5
    windows = sliding_window_view(data, window_size)
    print(f"\nSliding window (size={window_size}):")
    print(f"  Windows shape: {windows.shape}")
    print(f"  First window: {windows[0]}")

    # Peak detection
    peaks = find_peaks(data, height=0.5, distance=5)
    print(f"\nPeak detection:")
    print(f"  Peak indices: {peaks}")
    print(f"  Peak values: {data[peaks]}")

if __name__ == "__main__":
    main()