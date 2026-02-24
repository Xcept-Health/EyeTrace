# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt, acos

def angular_velocity_cy(double[:] v1, double[:] v2, double dt):
    cdef double dot = 0.0, norm1 = 0.0, norm2 = 0.0, cos_angle, angle
    cdef int i
    for i in range(3):
        dot += v1[i] * v2[i]
        norm1 += v1[i] * v1[i]
        norm2 += v2[i] * v2[i]
    norm1 = sqrt(norm1)
    norm2 = sqrt(norm2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    cos_angle = dot / (norm1 * norm2)
    if cos_angle > 1.0: cos_angle = 1.0
    elif cos_angle < -1.0: cos_angle = -1.0
    angle = acos(cos_angle)
    return angle / dt if dt > 0 else 0.0

def saccade_velocity_cy(double[:, ::1] gaze_positions, double[:] times):
    cdef:
        int n = gaze_positions.shape[0]
        int i
        double dt
        cnp.ndarray[double, ndim=1] vel = np.zeros(n, dtype=np.float64)
    
    if n < 2:
        return vel
    
    for i in range(1, n):
        dt = times[i] - times[i-1]
        vel[i-1] = angular_velocity_cy(gaze_positions[i-1], gaze_positions[i], dt)
    vel[n-1] = vel[n-2]  # duplicate last
    return vel

def detect_saccades_cy(double[:, ::1] gaze_positions, double[:] times,
                        double velocity_threshold, double min_duration):
    """
    Version Cython de détection des saccades (sans merging).
    Retourne un tableau de booléens indiquant les frames en saccade.
    """
    cdef:
        int n = gaze_positions.shape[0]
        int i
        double dt
        double vel
        cnp.ndarray[cnp.int8_t, ndim=1] is_saccade = np.zeros(n, dtype=np.int8)
        int in_saccade = 0
        int start_idx = 0
    
    if n < 2:
        return is_saccade
    
    for i in range(1, n):
        dt = times[i] - times[i-1]
        vel = angular_velocity_cy(gaze_positions[i-1], gaze_positions[i], dt)
        if vel > velocity_threshold:
            if not in_saccade:
                in_saccade = 1
                start_idx = i-1
        else:
            if in_saccade:
                # Check duration
                duration = times[i-1] - times[start_idx]
                if duration >= min_duration:
                    for j in range(start_idx, i):
                        is_saccade[j] = 1
                in_saccade = 0
    # Handle last saccade if still in
    if in_saccade:
        duration = times[n-1] - times[start_idx]
        if duration >= min_duration:
            for j in range(start_idx, n):
                is_saccade[j] = 1
    return is_saccade