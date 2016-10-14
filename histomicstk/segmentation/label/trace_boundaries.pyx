"""
Cython source code: trace_boundaries
"""
import cython

import numpy as np
cimport numpy as np

from libcpp.vector cimport vector

cdef extern from "trace_boundaries_opt.h":
  vector[vector[vector[int]]] trace_boundary(vector[vector[int]] label, int)


def trace_boundaries(np.ndarray[int, ndim=2, mode="c"] imLabel not None, connectivity=4):

    cdef vector[vector[vector[int]]] output

    output = trace_boundary(imLabel, connectivity)

    return output
