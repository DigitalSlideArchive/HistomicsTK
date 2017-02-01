import cython

import numpy as np
cimport numpy as np
cimport trace_boundaries_opt
from libcpp.vector cimport vector


def trace_boundaries_cython(
    np.ndarray[np.int64_t, ndim=2, mode="c"] im_label not None,
    connectivity=4, trace_all=False, 
    x_start=None, y_start=None, max_length=None):

    cdef vector[vector[vector[int]]] output

    if max_length is None:
        max_length = float('inf')

    if trace_all:

        output = trace_boundaries_opt.trace_label(
            im_label, connectivity, max_length)

    else:

        if x_start is None and y_start is None:

            output = trace_boundaries_opt.trace_boundary(
                im_label, connectivity, max_length)

        elif x_start is not None and y_start is not None:

            output = trace_boundaries_opt.trace_boundary(
                im_label, connectivity, max_length, x_start, y_start)

        else:

            raise ValueError("x_start or y_start is not defined !!")

    return output
