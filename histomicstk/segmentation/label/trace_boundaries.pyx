"""
Cython source code: trace_boundaries.pyx
"""
import cython

import numpy as np
cimport numpy as np
cimport trace_boundaries_opt
from libcpp.vector cimport vector


def trace_boundaries(np.ndarray[np.int64_t, ndim=2, mode="c"] imLabel not None, Connectivity=4, Label=False, XStart=None, YStart=None, MaxLength=None):

    cdef vector[vector[vector[int]]] output
    inf = MaxLength if MaxLength != None else float('inf')
    if Label:
        output = trace_boundaries_opt.trace_label(imLabel, Connectivity, inf)
    else:
        if XStart is None and YStart is None:
            output = trace_boundaries_opt.trace_boundary(imLabel, Connectivity, inf)
        elif XStart is not None and YStart is not None:
            output = trace_boundaries_opt.trace_boundary(imLabel, Connectivity, inf, XStart, YStart)
        else:
            raise ValueError("XStart or YStart is not defined !!")

    return output
