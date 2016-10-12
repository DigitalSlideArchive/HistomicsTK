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
        void rot90(int, int, vector[vector[int]] input,
                   vector[vector[int]] &output)
        vector[vector[vector[int]]] trace_boundary(vector[vector[int]] label, int)
        vector[vector[int]] isbf(int, int, vector[vector[int]] mask, int, int, float)
        vector[vector[int]] moore(int, int, vector[vector[int]] mask, int, int, float)


def trace_object_boundary(np.ndarray[int, ndim=2, mode="c"] imLabel not None, connectivity):

    cdef trace_boundary_cpp res
    res = trace_boundary_cpp()
    cdef vector[vector[vector[int]]] output

    output = res.trace_boundary(imLabel, connectivity)

    return output
