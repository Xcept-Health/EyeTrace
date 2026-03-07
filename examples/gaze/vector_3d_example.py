"""
Example: Estimate 3D gaze vector from eye landmarks (placeholder).
"""

import numpy as np
from eyetrace.gaze.vector_3d import gaze_vector_3d

def main():
    # Simulated eye landmarks (6 points)
    eye_landmarks = np.random.rand(6, 2) * 100
    gaze = gaze_vector_3d(eye_landmarks)
    print(f"Estimated gaze vector: {gaze}")
    # Should be [0,0,1] as placeholder

if __name__ == "__main__":
    main()