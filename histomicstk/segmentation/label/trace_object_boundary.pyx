# distutils: language = c++
# distutils: sources = trace_boundary_cpp.cpp
# distutils: extra_compile_args = -std=c++11

"""
Cython source code: trace_object_boundary
"""
import cython

import numpy as np
cimport numpy as np

from libcpp.vector cimport vector

cdef extern from "trace_boundary_cpp.h":
    cdef cppclass trace_boundary_cpp:
        trace_boundary_cpp() except +
        void rot90(int, int, vector[vector[int]] matrix,
                   vector[vector[int]] &matrix270,
                   vector[vector[int]] &matrix180,
                   vector[vector[int]] &matrix90)
        vector[vector[int]] isbf(int, int, vector[vector[int]] mask, int, int, float)
        vector[vector[int]] moore(int, int, vector[vector[int]] mask, int, int, float)

def trace_object_boundary(np.ndarray[int, ndim=2, mode="c"] mask not None, connectivity, xstart, ystart, INFINITY):
    cdef int m, n
    m, n = mask.shape[0], mask.shape[1]
    inf = INFINITY if INFINITY != 0 else float('inf')
    cdef trace_boundary_cpp res
    res = trace_boundary_cpp()
    cdef vector[vector[int]] boundary

    if connectivity == 4:
        boundary= res.isbf(m, n, mask, xstart, ystart, inf)
    elif connectivity == 8:
        boundary = res.moore(m, n, mask, xstart, ystart, inf)
    else:
        raise ValueError("Input 'Connectivity' must be 4 or 8.")

    cdef int size
    size = boundary[0].size()
    cdef np.ndarray h = np.zeros([2, size], dtype=np.int)
    cdef int i
    for i in range(size):
      h[0,i] = boundary[0][i]
      h[1,i] = boundary[1][i]
    return h[0], h[1]
