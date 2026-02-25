"""
Neck flexion angle estimation.
"""

import numpy as np

def neck_flexion_angle(face_landmarks, image_width, image_height,
                       shoulder_landmarks=None, camera_matrix=None):
    """
    Estimate neck flexion angle.

    The angle is computed as the pitch of the head relative to the vertical axis,
    optionally corrected by the orientation of the shoulders. If shoulder landmarks
    are not provided, the vertical is assumed to be the world vertical (image y-axis).

    Parameters
    ----------
    face_landmarks : mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList
        Face landmarks from MediaPipe Face Mesh.
    image_width, image_height : int
        Dimensions of the image.
    shoulder_landmarks : mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList, optional
        Pose landmarks from MediaPipe Pose. Should contain landmarks for left and right shoulders.
    camera_matrix : np.ndarray, shape (3,3), optional
        Intrinsic camera matrix. If None, a simple estimate is used.

    Returns
    -------
    float
        Neck flexion angle in radians. Positive = flexion (head down), negative = extension (head up).
    """
    # Estimate head pose angles
    from .angles import head_pose_angles
    pitch, roll, yaw = head_pose_angles(face_landmarks, image_width, image_height,
                                        camera_matrix=camera_matrix)

    # For now, just return the pitch as neck flexion (assuming neutral shoulders)
    # In a more advanced version, shoulder orientation could be used to correct.
    return pitch