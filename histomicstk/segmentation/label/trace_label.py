import numpy as np
import ctypes
from trace_object_boundary import trace_object_boundary


def trace_label(imLabel, Connectivity=8):

    imLabel = np.ascontiguousarray(imLabel, dtype=ctypes.c_int)

    output = trace_object_boundary(imLabel, Connectivity)

    return output
