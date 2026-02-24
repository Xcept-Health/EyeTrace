# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt

def eye_aspect_ratio(double[:, ::1] eye_landmarks):
    """
    Fast Cython implementation of EAR.
    """
    cdef:
        double vertical_1, vertical_2, horizontal
        double dx, dy

    # Vertical distance 1: p2 - p6
    dx = eye_landmarks[1,0] - eye_landmarks[5,0]
    dy = eye_landmarks[1,1] - eye_landmarks[5,1]
    vertical_1 = sqrt(dx*dx + dy*dy)

    # Vertical distance 2: p3 - p5
    dx = eye_landmarks[2,0] - eye_landmarks[4,0]
    dy = eye_landmarks[2,1] - eye_landmarks[4,1]
    vertical_2 = sqrt(dx*dx + dy*dy)

    # Horizontal distance: p1 - p4
    dx = eye_landmarks[0,0] - eye_landmarks[3,0]
    dy = eye_landmarks[0,1] - eye_landmarks[3,1]
    horizontal = sqrt(dx*dx + dy*dy)

    if horizontal == 0.0:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)