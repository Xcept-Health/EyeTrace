"""
Example: Compute mutual information between two related signals.
"""

import numpy as np
import matplotlib.pyplot as plt
from eyetrace.signal_analysis.mutual_info import mutual_information

def main():
    np.random.seed(0)
    n = 1000
    x = np.random.randn(n)
    # y = x + noise (high mutual info)
    y = x + 0.5 * np.random.randn(n)
    # z = independent noise (low mutual info)
    z = np.random.randn(n)

    mi_xy = mutual_information(x, y, bins=20)
    mi_xz = mutual_information(x, z, bins=20)

    print(f"Mutual information I(x;y) = {mi_xy:.4f} nats")
    print(f"Mutual information I(x;z) = {mi_xz:.4f} nats")

    # Scatter plots
    plt.figure(figsize=(12, 5))
    plt.subplot(1,2,1); plt.scatter(x, y, s=1); plt.title(f'x vs y (MI={mi_xy:.3f})')
    plt.subplot(1,2,2); plt.scatter(x, z, s=1); plt.title(f'x vs z (MI={mi_xz:.3f})')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()