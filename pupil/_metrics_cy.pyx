# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt

def variance(double[:] diameters):
    cdef:
        int n = diameters.shape[0]
        double mean = 0.0
        double var = 0.0
        int i

    if n < 2:
        return 0.0

    # First pass: compute mean
    for i in range(n):
        mean += diameters[i]
    mean /= n

    # Second pass: compute variance (unbiased)
    for i in range(n):
        var += (diameters[i] - mean) * (diameters[i] - mean)
    var /= (n - 1)

    return var


def std_dev(double[:] diameters):
    cdef double var = variance(diameters)
    return sqrt(var)


def coefficient_variation(double[:] diameters):
    cdef:
        int n = diameters.shape[0]
        double mean = 0.0
        double var = 0.0
        double std
        int i

    if n < 2:
        return 0.0

    for i in range(n):
        mean += diameters[i]
    mean /= n

    if mean == 0.0:
        return 0.0

    for i in range(n):
        var += (diameters[i] - mean) * (diameters[i] - mean)
    var /= (n - 1)
    std = sqrt(var)

    return (std / mean) * 100.0


def zscore(double[:] diameters):
    cdef:
        int n = diameters.shape[0]
        double mean = 0.0
        double var = 0.0
        double std
        int i
        cnp.ndarray[double, ndim=1] result = np.empty(n, dtype=np.float64)

    if n == 0:
        return result

    for i in range(n):
        mean += diameters[i]
    mean /= n

    for i in range(n):
        var += (diameters[i] - mean) * (diameters[i] - mean)
    var /= (n - 1) if n > 1 else 1.0
    std = sqrt(var)

    if std == 0.0:
        for i in range(n):
            result[i] = 0.0
    else:
        for i in range(n):
            result[i] = (diameters[i] - mean) / std

    return result