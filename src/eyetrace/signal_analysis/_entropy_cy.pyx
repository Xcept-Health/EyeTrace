# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
import numpy as np
cimport numpy as cnp
from libc.math cimport log, sqrt, NAN

def sample_entropy(double[:] data, int m, double r):
    """
    Sample entropy (SampEn) implemented in Cython.
    """
    cdef:
        int N = data.shape[0]
        int i, j, k
        int count, matches
        double sim, diff
        double B = 0.0, A = 0.0
        double mean = 0.0, var = 0.0, std

    if N < m + 1:
        return NAN

    # Compute standard deviation if r is given as factor (r == 0.0)
    if r == 0.0:
        for i in range(N):
            mean += data[i]
        mean /= N
        for i in range(N):
            var += (data[i] - mean) * (data[i] - mean)
        var /= (N - 1)
        std = sqrt(var)
        r = 0.2 * std

    # Count matches for m and m+1
    for i in range(N - m):
        # Count similar vectors of length m
        matches = 0
        for j in range(N - m):
            if j == i:
                continue
            sim = 1
            for k in range(m):
                diff = data[i+k] - data[j+k]
                if diff < 0:
                    diff = -diff
                if diff > r:
                    sim = 0
                    break
            if sim:
                matches += 1
        B += matches

        # Count similar vectors of length m+1
        matches = 0
        for j in range(N - m - 1):
            if j == i:
                continue
            sim = 1
            for k in range(m + 1):
                diff = data[i+k] - data[j+k]
                if diff < 0:
                    diff = -diff
                if diff > r:
                    sim = 0
                    break
            if sim:
                matches += 1
        A += matches

    if B == 0 or A == 0:
        return NAN
    return -log(A / B)