"""
Head pose angles (pitch, roll, yaw) using solvePnP.
"""

import numpy as np
import cv2
from .utils import get_face_landmark_array, FACE_MODEL_POINTS, FACE_MODEL_INDICES

def head_pose_angles(face_landmarks, image_width, image_height, camera_matrix=None, dist_coeffs=None):
    """
    Estimate head pose angles using Perspective-n-Point.

    Parameters
    ----------
    face_landmarks : mediapipe face landmarks
    image_width, image_height : int
        Image dimensions.
    camera_matrix : np.ndarray, shape (3,3), optional
        Intrinsic camera matrix. If None, a simple estimate is used.
    dist_coeffs : np.ndarray, optional
        Distortion coefficients.

    Returns
    -------
    pitch, roll, yaw : float
        Angles in radians.
    """
    # Get 2D points from landmarks
    points_2d = get_face_landmark_array(face_landmarks, FACE_MODEL_INDICES, image_width, image_height)

    # Camera matrix (if not provided, assume simple pinhole)
    if camera_matrix is None:
        focal_length = image_width  # rough estimate
        camera_matrix = np.array([
            [focal_length, 0, image_width/2],
            [0, focal_length, image_height/2],
            [0, 0, 1]
        ], dtype=np.float64)
    if dist_coeffs is None:
        dist_coeffs = np.zeros((4, 1))

    # Solve PnP
    success, rvec, tvec = cv2.solvePnP(
        FACE_MODEL_POINTS,
        points_2d,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return np.nan, np.nan, np.nan

    # Convert rotation vector to rotation matrix
    rmat, _ = cv2.Rodrigues(rvec)

    # Decompose rotation matrix to Euler angles (pitch, roll, yaw)
    # Using the convention: pitch = atan2(-r[2,0], sqrt(r[0,0]^2 + r[1,0]^2)), etc.
    # For a more standard aerospace sequence, we can use cv2.RQDecomp3x3.
    angles = cv2.RQDecomp3x3(rmat)[0]  # returns (pitch, roll, yaw) in degrees? Actually returns Euler angles in degrees.
    # Convert to radians
    pitch = np.radians(angles[0])
    roll = np.radians(angles[1])
    yaw = np.radians(angles[2])

    return pitch, roll, yaw