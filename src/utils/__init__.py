"""
Utility functions for EyeTrace.

This module provides helper functions for landmark extraction,
filtering, geometry, and general math operations.
"""

from .landmarks import (
    extract_eye_landmarks_from_mediapipe,
    extract_iris_landmarks_from_mediapipe,
    extract_face_landmarks_array
)
from .filtering import (
    moving_average,
    savgol_filter,
    kalman_filter_1d
)
from .geometry import (
    angle_between_vectors,
    distance,
    normalize_vector,
    project_point_to_line
)
from .math_helpers import (
    mad_outlier_removal,
    sliding_window_view,
    find_peaks
)

__all__ = [
    # landmarks
    'extract_eye_landmarks_from_mediapipe',
    'extract_iris_landmarks_from_mediapipe',
    'extract_face_landmarks_array',
    # filtering
    'moving_average',
    'savgol_filter',
    'kalman_filter_1d',
    # geometry
    'angle_between_vectors',
    'distance',
    'normalize_vector',
    'project_point_to_line',
    # math_helpers
    'mad_outlier_removal',
    'sliding_window_view',
    'find_peaks',
]