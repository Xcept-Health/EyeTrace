# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt

def first_derivative(double[:] diameters, double dt=1.0):
    """
    Compute first derivative using central differences.
    """
    cdef:
        int n = diameters.shape[0]
        int i
        cnp.ndarray[double, ndim=1] deriv = np.empty(n, dtype=np.float64)

    if n < 2:
        for i in range(n):
            deriv[i] = 0.0
        return deriv

    # Forward difference for first point
    deriv[0] = (diameters[1] - diameters[0]) / dt

    # Central differences for interior points
    for i in range(1, n-1):
        deriv[i] = (diameters[i+1] - diameters[i-1]) / (2.0 * dt)

    # Backward difference for last point
    deriv[n-1] = (diameters[n-1] - diameters[n-2]) / dt

    return deriv