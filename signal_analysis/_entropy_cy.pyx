# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
import numpy as np
cimport numpy as cnp
from libc.math cimport log, fabs

def sample_entropy(double[:] signal, int m=2, double r=0.0):
    """
    Fast Cython implementation of Sample Entropy.
    """
    cdef:
        int n = signal.shape[0]
        int i, j, k
        int count_m, count_mp1
        double maxdist, diff
        double Bm, Am

    if n < m + 1:
        return np.nan

    if r == 0.0:
        # Estimate r as 0.2 * std if not provided
        # Need to compute std first
        cdef double mean = 0.0, var = 0.0
        for i in range(n):
            mean += signal[i]
        mean /= n
        for i in range(n):
            var += (signal[i] - mean) * (signal[i] - mean)
        var /= (n - 1)
        r = 0.2 * sqrt(var)

    cdef int Nm = n - m + 1
    cdef int Nm1 = n - m

    # Build templates implicitly (no need to store all)
    Bm = 0.0
    for i in range(Nm):
        for j in range(i+1, Nm):
            maxdist = 0.0
            for k in range(m):
                diff = fabs(signal[i+k] - signal[j+k])
                if diff > maxdist:
                    maxdist = diff
            if maxdist <= r:
                Bm += 2.0  # symmetric count
    Bm /= (Nm * (Nm - 1))

    Am = 0.0
    for i in range(Nm1):
        for j in range(i+1, Nm1):
            maxdist = 0.0
            for k in range(m+1):
                diff = fabs(signal[i+k] - signal[j+k])
                if diff > maxdist:
                    maxdist = diff
            if maxdist <= r:
                Am += 2.0
    Am /= (Nm1 * (Nm1 - 1))

    if Bm == 0.0 or Am == 0.0:
        return np.nan
    return -log(Am / Bm)