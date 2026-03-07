"""
Example: Compute pupil eccentricity from pupil and iris centers.
"""

import numpy as np
from eyetrace.gaze.eccentricity import pupil_eccentricity

def main():
    # Simulated pupil and iris centers (2D)
    pupil_center = np.array([320, 240])
    iris_center = np.array([318, 243])
    # Dummy gaze vector (not used in current implementation)
    gaze_vector = np.array([0, 0, 1])

    eccentricity = pupil_eccentricity(pupil_center, iris_center, gaze_vector)
    print(f"Pupil eccentricity (2D distance): {eccentricity:.2f} pixels")

    # Simulated 3D centers (arbitrary units)
    pupil_3d = np.array([0, 0, 0])
    iris_3d = np.array([0.1, 0.05, 0.0])
    eccentricity_3d = pupil_eccentricity(pupil_3d, iris_3d, gaze_vector)
    print(f"Pupil eccentricity (3D angle): {np.degrees(eccentricity_3d):.2f}°")

if __name__ == "__main__":
    main()