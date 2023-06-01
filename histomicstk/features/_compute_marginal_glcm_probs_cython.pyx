import numpy as np

cimport cython
cimport numpy as np


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def _compute_marginal_glcm_probs_cython(double[:, :] nGLCM not None):

    cdef long num_levels = nGLCM.shape[0]

    cdef double[::1] px = np.zeros([num_levels], dtype=float)
    cdef double[::1] py = np.zeros([num_levels], dtype=float)

    cdef double[::1] pxPlusy = np.zeros([2*num_levels-1], dtype=float)
    cdef double[::1] pxMinusy = np.zeros([num_levels], dtype=float)
    cdef int i, j, i_minus_j

    with nogil:
        for i in range(num_levels):
            for j in range(num_levels):
                px[i] = px[i] + nGLCM[i, j]
                py[i]= py[i] + nGLCM[j, i]
                pxPlusy[i+j] = pxPlusy[i+j] + nGLCM[i, j]
                i_minus_j = i-j
                if i < j:
                    i_minus_j *= -1
                pxMinusy[i_minus_j] = pxMinusy[i_minus_j] + nGLCM[i, j]

    return np.asarray(px), np.asarray(py), np.asarray(pxPlusy), np.asarray(pxMinusy)
