"""
Cython source code: trace_boundaries_opt.pxd
"""
from libcpp.vector cimport vector


cdef extern from "trace_boundaries_opt.hpp":
  vector[vector[vector[int]]] trace_boundary(vector[vector[int]] label, int Connectivity, float inf, int XStart, int YStart)
  vector[vector[vector[int]]] trace_boundary(vector[vector[int]] label, int Connectivity, float inf)