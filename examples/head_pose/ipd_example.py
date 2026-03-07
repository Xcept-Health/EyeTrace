"""
Example: Compute inter-pupillary distance from simulated pupil centers.
"""

import numpy as np
from eyetrace.head_pose.ipd import inter_pupillary_distance

def main():
    # Simulated pupil centers (pixels)
    left_pupil = np.array([300, 240])
    right_pupil = np.array([340, 242])

    ipd = inter_pupillary_distance(left_pupil, right_pupil)
    print(f"Inter-pupillary distance: {ipd:.2f} pixels")

    # If we know pixel-to-mm conversion, we can convert
    px_to_mm = 0.05  # example
    ipd_mm = ipd * px_to_mm
    print(f"IPD in mm: {ipd_mm:.2f} mm")

if __name__ == "__main__":
    main()