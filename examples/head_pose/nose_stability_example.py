"""
Example: Compute nose tip stability over time.
"""

import numpy as np
from eyetrace.head_pose.nose_stability import nose_stability

class SimulatedLandmarkSequence:
    def __init__(self, num_frames, img_w, img_h, noise_std=2):
        self.frames = []
        x, y = img_w // 2, img_h // 2
        for _ in range(num_frames):
            # Add random walk
            x += np.random.randn() * noise_std
            y += np.random.randn() * noise_std
            # Normalize
            norm_x = x / img_w
            norm_y = y / img_h
            # Create a full landmarks object with 468 points, but only nose tip (index 1) is set
            lm_list = [type('Landmark', (), {'x': 0, 'y': 0, 'z': 0}) for _ in range(468)]
            lm_list[1] = type('Landmark', (), {'x': norm_x, 'y': norm_y, 'z': 0})
            # Wrap in a face_landmarks-like object
            face_lm = type('FaceLandmarks', (), {'landmark': lm_list})()
            self.frames.append(face_lm)

def main():
    img_w, img_h = 640, 480
    seq = SimulatedLandmarkSequence(100, img_w, img_h, noise_std=5)
    variance = nose_stability(seq.frames, img_w, img_h)
    print(f"Nose tip variance: {variance:.2f} pixels²")

if __name__ == "__main__":
    main()