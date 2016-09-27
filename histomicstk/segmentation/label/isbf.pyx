"""
Cython source code: ISBF for trace_boundary
"""
import cython

import numpy as np
cimport numpy as np

from libcpp.vector cimport vector

cdef extern from "isbfcpp.h":
    cdef cppclass isbfcpp:
        isbfcpp() except +
        void rot90(int, int, vector[vector[int]] matrix,
                   vector[vector[int]] &matrix270,
                   vector[vector[int]] &matrix180,
                   vector[vector[int]] &matrix90)
        vector[vector[int]] traceBoundary(int, int, vector[vector[int]] mask, int, int, float)

def isbf(np.ndarray[int, ndim=2, mode="c"] mask not None, xstart, ystart, INFINITY):
    cdef int m, n
    m, n = mask.shape[0], mask.shape[1]
    inf = INFINITY if INFINITY != 0 else float('inf')
    cdef isbfcpp res
    res = isbfcpp()
    cdef vector[vector[int]] boundary = res.traceBoundary(m, n, mask, xstart, ystart, inf)
    cdef int size
    size = boundary[0].size()
    cdef np.ndarray h = np.zeros([2, size], dtype=np.int)
    cdef int i
    for i in range(size):
      h[0,i] = boundary[0][i]
      h[1,i] = boundary[1][i]
    return h[0], h[1]
