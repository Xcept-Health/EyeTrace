"""
Head pose and face analysis module.

Provides functions to estimate head orientation (pitch, roll, yaw),
mouth aspect ratio (MAR), yawning detection, nose stability, neck flexion,
inter-pupillary distance (IPD), and postural sag.
"""

from .angles import head_pose_angles
from .angular_velocity import head_angular_velocity
from .mar import mouth_aspect_ratio
from .yawning import yawn_detection, yawn_frequency
from .nose_stability import nose_stability
from .neck_angle import neck_flexion_angle
from .ipd import inter_pupillary_distance
from .postural_sag import postural_sag

__all__ = [
    'head_pose_angles',
    'head_angular_velocity',
    'mouth_aspect_ratio',
    'yawn_detection',
    'yawn_frequency',
    'nose_stability',
    'neck_flexion_angle',
    'inter_pupillary_distance',
    'postural_sag',
]