"""
Cython source code: ISBF for TraceBounds
"""
import cython

import numpy as np
cimport numpy as np

from libcpp cimport bool
from libcpp.vector cimport vector

cdef extern from "passInf.h":
    pass

cdef extern from "isbfcpp.h":
    cdef cppclass isbfcpp:
        isbfcpp() except +
        int length()
        void nth_from_last(int, int &x, int &y)
        bool addList(int, int)
        void roateMatrix(int, int, int **input, int **output)
        vector[int] getList(int, int, int *size, int *mask, int, int, float)
        void clean()

def isbf(np.ndarray[int, ndim=2, mode="c"] mask not None, xstart, ystart, INFINITY):
    cdef int m, n
    cdef int size
    m, n = mask.shape[0], mask.shape[1]
    inf = INFINITY if INFINITY != 0 else float('inf')
    cdef isbfcpp res
    res = isbfcpp()
    cdef vector[int] boundary = res.getList(m, n, &size, &mask[0,0], xstart, ystart, inf)
    cdef np.ndarray h = np.zeros([2, size-1], dtype=np.int)
    cdef int i
    for i in range(size-1):
      h[0,i] = boundary[i]
      h[1,i] = boundary[i+size-1]
    return h[0], h[1]
