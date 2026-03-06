import unittest
import numpy as np
from eyetrace.eyelids.utils import ( 
    extract_eye_landmarks_from_mediapipe,
    extract_both_eyes,
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES
)

# Mock classes to simulate MediaPipe structures
class MockLandmark:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z

class MockFaceLandmarks:
    def __init__(self, landmarks):
        self.landmark = landmarks

class TestEyeLandmarkExtraction(unittest.TestCase):
    def setUp(self):
        # Create 400 mock landmarks with unique normalized coordinates
        # We'll use x = i/400, y = (i % 100)/100 to have some variation
        self.num_landmarks = 400
        self.landmarks = [
            MockLandmark(x=i / self.num_landmarks, y=(i % 100) / 100.0)
            for i in range(self.num_landmarks)
        ]
        self.face_landmarks = MockFaceLandmarks(self.landmarks)
        self.image_width = 640
        self.image_height = 480

    def test_extract_left_eye_correct_indices(self):
        """Check that left eye indices are used and coordinates converted correctly."""
        left_eye = extract_eye_landmarks_from_mediapipe(
            self.face_landmarks, self.image_width, self.image_height, 'left'
        )
        self.assertEqual(left_eye.shape, (6, 2))
        for i, idx in enumerate(LEFT_EYE_INDICES):
            lm = self.landmarks[idx]
            expected_x = int(lm.x * self.image_width)
            expected_y = int(lm.y * self.image_height)
            np.testing.assert_array_equal(left_eye[i], [expected_x, expected_y])

    def test_extract_right_eye_correct_indices(self):
        """Check that right eye indices are used and coordinates converted correctly."""
        right_eye = extract_eye_landmarks_from_mediapipe(
            self.face_landmarks, self.image_width, self.image_height, 'right'
        )
        self.assertEqual(right_eye.shape, (6, 2))
        for i, idx in enumerate(RIGHT_EYE_INDICES):
            lm = self.landmarks[idx]
            expected_x = int(lm.x * self.image_width)
            expected_y = int(lm.y * self.image_height)
            np.testing.assert_array_equal(right_eye[i], [expected_x, expected_y])

    def test_extract_both_eyes(self):
        """Check that both eyes are returned correctly."""
        left, right = extract_both_eyes(
            self.face_landmarks, self.image_width, self.image_height
        )
        # Verify left eye
        self.assertEqual(left.shape, (6, 2))
        for i, idx in enumerate(LEFT_EYE_INDICES):
            lm = self.landmarks[idx]
            expected = [int(lm.x * self.image_width), int(lm.y * self.image_height)]
            np.testing.assert_array_equal(left[i], expected)
        # Verify right eye
        self.assertEqual(right.shape, (6, 2))
        for i, idx in enumerate(RIGHT_EYE_INDICES):
            lm = self.landmarks[idx]
            expected = [int(lm.x * self.image_width), int(lm.y * self.image_height)]
            np.testing.assert_array_equal(right[i], expected)

    def test_invalid_eye_raises_error(self):
        """Passing an invalid eye string should raise a ValueError."""
        with self.assertRaises(ValueError):
            extract_eye_landmarks_from_mediapipe(
                self.face_landmarks, self.image_width, self.image_height, 'invalid'
            )

if __name__ == '__main__':
    unittest.main()