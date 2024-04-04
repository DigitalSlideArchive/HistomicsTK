import numpy as np

cimport cython
cimport numpy as np


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def _max_clustering_cython(double[:, :] im not None, int[:, :] im_fgnd_mask not None, double rad):

    cdef long sx = im.shape[1]
    cdef long sy = im.shape[0]
    cdef long r = int(rad + 0.5)

    # find foreground pixels
    cdef long[:] px, py

    py, px = np.nonzero(im_fgnd_mask)

    cdef long num_pixels = py.shape[0]

    # find local maximum of all requested pixels
    cdef double[:, ::1] local_max_val = np.zeros([sy, sx], dtype=float)
    cdef long[:, ::1] local_max_ind = np.zeros([sy, sx], dtype=np.int64)
    cdef int[:, ::1] peak_found = np.zeros([sy, sx], dtype=np.int32)
    cdef double min_im_val = np.min(im)

    cdef long ox, oy, mx, my, cx, cy, nx, ny
    cdef double mval, cval, nval
    cdef int changed

    with nogil:
        for i in range(num_pixels):

            cx = px[i]

            if cx < 0 or cx >= sx:
                continue

            cy = py[i]

            if cy < 0 or cy >= sy:
                continue

            cval = im[cy, cx]

            my = cy
            mx = cx
            mval = cval

            changed = 0

            for ox in range(-r, r+1):

                nx = cx + ox

                if nx < 0 or nx >= sx:
                    continue

                for oy in range(-r, r+1):

                    if (ox * ox + oy * oy) > rad * rad:
                        continue

                    ny = cy + oy

                    if ny < 0 or ny >= sy:
                        continue

                    nval = min_im_val

                    if im_fgnd_mask[ny, nx]:
                        nval = im[ny, nx]

                    if nval > mval:

                        changed = True
                        mval = nval
                        mx = nx
                        my = ny

            local_max_val[cy, cx] = mval
            local_max_ind[cy, cx] = my * sx + mx

            if not changed:  # this pixel itself is the maximum in its neighborhood
                peak_found[cy, cx] = 1

    # find local peaks of all requested pixels
    cdef np.ndarray[np.int64_t, ndim=2, mode='c'] maxpath_np = np.zeros([1000, 2], dtype=np.int64)
    cdef long[:, ::1] maxpath = maxpath_np
    cdef long end_x, end_y, end_pos, end_ind, end_max_ind
    cdef long path_len = maxpath.shape[0]

    with nogil:
        for i in range(num_pixels):

            cx = px[i]

            if cx < 0 or cx >= sx:
                continue

            cy = py[i]

            if cy < 0 or cy >= sy:
                continue

            # initialize tracking trajectory
            end_pos = 0

            end_x = cx
            end_y = cy
            end_ind = cy * sx + cx
            end_max_ind = local_max_ind[end_y, end_x]

            maxpath[end_pos, 0] = end_x
            maxpath[end_pos, 1] = end_y

            while not peak_found[end_y, end_x]:

                # increment trajectory counter
                end_pos += 1

                # if overflow, increase size
                if end_pos >= path_len:
                    with gil:
                        maxpath_np.resize([path_len * 2, 2])
                    maxpath = maxpath_np
                    path_len = maxpath.shape[0]

                # add local max to trajectory
                end_ind = end_max_ind
                end_x = end_ind % sx
                end_y = end_ind / sx
                end_max_ind = local_max_ind[end_y, end_x]

                maxpath[end_pos, 0] = end_x
                maxpath[end_pos, 1] = end_y

            for i in range(end_pos+1):

                cx = maxpath[i, 0]
                cy = maxpath[i, 1]

                local_max_ind[cy, cx] = end_max_ind
                local_max_val[cy, cx] = local_max_val[end_y, end_x]
                peak_found[cy, cx] = 1

    return np.asarray(local_max_val), np.asarray(local_max_ind)
