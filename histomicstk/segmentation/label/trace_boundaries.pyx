"""
Cython source code: trace_boundaries
"""
import cython

import numpy as np
cimport numpy as np

from libcpp.vector cimport vector


cdef extern from "trace_boundaries_opt.h":
    cdef cppclass trace_boundaries_opt:
        trace_boundaries_opt() except +
        void rot90(int, int, vector[vector[int]] input,
                   vector[vector[int]] &output)
        vector[vector[vector[int]]] trace_boundary(vector[vector[int]] label, int)
        vector[vector[int]] isbf(int, int, vector[vector[int]] mask, int, int, float)
        vector[vector[int]] moore(int, int, vector[vector[int]] mask, int, int, float)


def trace_boundaries(np.ndarray[int, ndim=2, mode="c"] imLabel not None, connectivity=4):

    cdef trace_boundaries_opt res
    res = trace_boundaries_opt()
    cdef vector[vector[vector[int]]] output

    output = res.trace_boundary(imLabel, connectivity)

    return output
