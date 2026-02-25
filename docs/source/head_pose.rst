.. _head_pose:

Head Pose Module
================

.. automodule:: eyetrace.head_pose
   :members:
   :undoc-members:
   :show-inheritance:

The head pose module provides functions to estimate head orientation (pitch, roll, yaw),
compute angular velocity, detect yawning via Mouth Aspect Ratio (MAR), measure nose
stability, estimate neck flexion angle, calculate inter-pupillary distance (IPD),
and track postural sag over time.

.. toctree::
   :maxdepth: 1
   :caption: Submodules

   head_pose.angles
   head_pose.angular_velocity
   head_pose.mar
   head_pose.yawning
   head_pose.nose_stability
   head_pose.neck_angle
   head_pose.ipd
   head_pose.postural_sag