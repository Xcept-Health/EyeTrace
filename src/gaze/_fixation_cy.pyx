# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt

def fixation_dispersion_cy(double[:, ::1] points):
    """
    Compute mean standard deviation of points (2D or 3D).
    points : shape (n, d) avec d=2 ou 3.
    """
    cdef:
        int n = points.shape[0]
        int d = points.shape[1]
        int i, j
        double[:] mean = np.zeros(d, dtype=np.float64)
        double[:] var = np.zeros(d, dtype=np.float64)
        double disp = 0.0
    
    if n < 2:
        return 0.0
    
    # Compute mean
    for j in range(d):
        for i in range(n):
            mean[j] += points[i, j]
        mean[j] /= n
    
    # Compute variance
    for j in range(d):
        for i in range(n):
            var[j] += (points[i, j] - mean[j]) * (points[i, j] - mean[j])
        var[j] /= n  # population variance
        disp += sqrt(var[j])
    
    return disp / d  # mean of std deviations