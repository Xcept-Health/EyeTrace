# cython: boundscheck=False, wraparound=False
import numpy as np
cimport numpy as cnp
from libc.math cimport log2

def lempel_ziv_complexity(double[:] signal, int normalize=True):
    cdef:
        int n = signal.shape[0]
        int i
        double median
        # Pour la binarisation, on peut trier pour trouver la médiane (simple)
    # Version simplifiée : on utilise numpy pour la médiane, puis on convertit en binaire
    # Pour une version pure Cython, il faudrait implémenter un quickselect, mais on peut
    # laisser numpy gérer la médiane, ce n'est pas critique.
    # On va faire une version mixte : calcul de la médiane en Python, puis boucle Cython.
    # À des fins de démo, je vais laisser la médiane en numpy.
    # Si vraiment performance requise, on pourrait passer directement la séquence binaire.
    import numpy as np
    median = np.median(np.asarray(signal))
    cdef int[:] binary = (np.asarray(signal) > median).astype(np.int32)

    cdef int c = 1
    cdef int u = 1
    cdef int v = 1
    cdef int v_max = 1

    for i in range(1, n):
        if binary[i] == binary[v-1]:
            v += 1
        else:
            if v > v_max:
                v_max = v
            v = i - u + 1
            u = i + 1
            c += 1
    if v > v_max:
        v_max = v
    c += 1

    if normalize:
        return c * log2(<double>n) / n
    else:
        return <double>c