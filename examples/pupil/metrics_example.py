"""
Example: Basic statistical metrics for pupil diameter.
"""

import numpy as np
from eyetrace.pupil.metrics import (
    variance,
    std_dev,
    coefficient_variation,
    normalized_diameter,
    zscore
)

def main():
    # Simulate pupil data
    np.random.seed(42)
    diameters = 4.0 + 0.5 * np.random.randn(100)  # mean 4, std ~0.5

    var = variance(diameters)
    std = std_dev(diameters)
    cv = coefficient_variation(diameters)

    print(f"Variance: {var:.4f}")
    print(f"Standard deviation: {std:.4f}")
    print(f"Coefficient of variation: {cv:.2f}%")

    # Normalization
    baseline = np.mean(diameters[:20])  # first 20 as baseline
    norm = normalized_diameter(diameters, baseline)
    print(f"First 5 normalized values: {norm[:5]}")

    # Z-score
    z = zscore(diameters)
    print(f"First 5 z-scores: {z[:5]}")

if __name__ == "__main__":
    main()