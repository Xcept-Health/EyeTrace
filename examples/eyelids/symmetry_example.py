"""
Example: Compute eyelid symmetry (correlation between left and right EAR).
"""

import numpy as np
from eyetrace.eyelids.symmetry import eyelid_symmetry

def main():
    # Simulate left and right EAR with high symmetry
    np.random.seed(42)
    t = np.arange(0, 60, 0.01)
    base = 0.3 + 0.05 * np.sin(2 * np.pi * 0.1 * t)
    left_ear = base + 0.01 * np.random.randn(len(t))
    right_ear = base + 0.01 * np.random.randn(len(t))  # very similar

    corr = eyelid_symmetry(left_ear, right_ear)
    print(f"Symmetry (correlation) between left and right EAR: {corr:.3f}")

    # Simulate asymmetric (right eye affected)
    right_ear_asym = base + 0.1 * np.sin(2 * np.pi * 0.3 * t) + 0.02 * np.random.randn(len(t))
    corr_asym = eyelid_symmetry(left_ear, right_ear_asym)
    print(f"Symmetry with asymmetry: {corr_asym:.3f}")

if __name__ == "__main__":
    main()