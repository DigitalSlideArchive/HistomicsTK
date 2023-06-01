import numpy as np

cimport cython
cimport numpy as np
from libc.math cimport M_PI, cos, round, sin
from libcpp.vector cimport vector


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def _trace_object_boundaries_cython(long[:, :] im_label not None, long connectivity, long x_start, long y_start, double max_length):

    cdef long nrows = im_label.shape[0]
    cdef long ncols = im_label.shape[1]

    cdef int flag = 0

    cdef long[:, :] im_label_270 = _rot90(im_label)
    cdef long[:, :] im_label_180 = _rot90(im_label_270)
    cdef long[:, :] im_label_90 = _rot90(im_label_180)

    cdef long i, j, t
    with nogil:

        # find starting x and y points if not defined
        if (x_start == -1) & (y_start == -1):

          for i in range(nrows):
            for j in range(ncols):

              if (im_label[i, j] > 0) & (flag == 0):
                # check if the number of points is one
                if ((im_label[i, j+1] == 0) & (im_label[i+1, j] == 0) & (im_label[i+1, j+1] == 0) & (im_label[i-1, j+1] == 0)) == False:
                  x_start = j
                  y_start = i
                  flag = 1
                  break
            if flag == 1:
                break

    bx = []
    by = []

    if connectivity == 4:
        bx, by = _isbf(
            im_label, im_label_90, im_label_180, im_label_270,
            x_start, y_start, max_length);

    else:
        bx, by = _moore(
            im_label, im_label_90, im_label_180, im_label_270,
            x_start, y_start, max_length);

    return np.asarray(bx), np.asarray(by)


cdef long[:, :] _rot90(long[:, :] input):

    cdef long nrows = input.shape[0]
    cdef long ncols = input.shape[1]

    cdef long[:, :] output = np.zeros([ncols, nrows], dtype=int)

    cdef long i, j

    for i in range(nrows):
        for j in range(ncols):
          output[j, nrows-1-i] = input[i, j]

    return output


def _moore(long[:, :] mask, long[:, :] mask_90, long[:, :] mask_180, long[:, :] mask_270, int x_start, int y_start, float max_length):

    cdef long nrows = mask.shape[0]
    cdef long ncols = mask.shape[1]

    # initialize boundary vector
    cdef vector[long] list_bx
    cdef vector[long] list_by

    # push the first x and y points
    list_bx.push_back(x_start);
    list_by.push_back(y_start);

    # check degenerate case where mask contains 1 pixel
    cdef long sum = np.sum(mask)

    # set size of boundary points: the size of X is equal to the size of Y
    cdef long size_boundary

    cdef long[:, :] h
    cdef long a, b, x, y

    # initialize cX and cY which indicate directions for each Moore
    cdef long cX
    cdef long cY

    # set sin and cos
    cdef double s, c, p, q, ci, cj

    # initialize default direction
    cdef long DX = 1
    cdef long DY = 0

    # set the number of rows and cols for ISBF
    cdef long row_isbf = 3
    cdef long col_isbf = 3

    cdef double angle

    # define clockwise ordered indices
    cdef vector[long] row = [2, 1, 0, 0, 0, 1, 2, 2]
    cdef vector[long] col = [0, 0, 0, 1, 2, 2, 2, 1]
    cdef vector[long] dX = [-1, 0, 0, 1, 1, 0, 0, -1]
    cdef vector[long] dY = [0, -1, -1, 0, 0, 1, 1, 0]
    cdef vector[long] oX = [-1, -1, -1, 0, 1, 1, 1, 0]
    cdef vector[long] oY = [1, 0, -1, -1, -1, 0, 1, 1]

    cdef long move = 0
    cdef int is_moore = 0

    cdef long fx1, fx2, fy1, fy2
    cdef long lx1, lx2, ly1, ly2

    cdef long i, j

    if sum > 1:

      # loop until true
      while(True):

        h = np.zeros((row_isbf, col_isbf), dtype=int)

        with nogil:
            with cython.boundscheck(False):
                # initialize a and b which are indices of moore
                a = 0
                b = 0

                # get length of the current linked list
                size_boundary = list_bx.size()

                x = list_bx[size_boundary-1]
                y = list_by[size_boundary-1]

                if (DX == 1) & (DY == 0):
                  for i in range(ncols-x-2, ncols-x+1):
                    for j in range(y-1, y+2):
                        h[a, b] = mask_90[i, j]
                        b = b + 1
                    b = 0
                    a = a + 1
                  angle = M_PI/2

                elif (DX == 0) & (DY == -1):
                  for i in range(y-1, y+2):
                    for j in range(x-1, x+2):
                        h[a, b] = mask[i, j]
                        b = b + 1
                    b = 0
                    a = a + 1
                  angle = 0

                elif (DX == -1) & (DY == 0):
                  for i in range(x-1, x+2):
                    for j in range(nrows-y-2, nrows-y+1):
                        h[a, b] = mask_270[i, j]
                        b = b + 1
                    b = 0
                    a = a + 1
                  angle = 3*M_PI/2

                else:
                  for i in range(nrows-y-2, nrows-y+1):
                    for j in range(ncols-x-2, ncols-x+1):
                        h[a, b] = mask_180[i, j]
                        b = b + 1
                    b = 0
                    a = a + 1
                  angle = M_PI

                move = 0
                is_moore = 0

                for i in range(8):
                  if is_moore == 0:
                    if h[row[i], col[i]] == 1:
                        move = i
                        is_moore = 1

                cX = oX[move]
                cY = oY[move]
                DX = dX[move]
                DY = dY[move]

                # transform points by incoming directions and add to contours
                s = sin(angle)
                c = cos(angle)

                p = c*cX - s*cY
                q = s*cX + c*cY

        list_bx.push_back(list_bx.back()+int(round(p)))
        list_by.push_back(list_by.back()+int(round(q)))

        with nogil:
            ci = c*DX-s*DY
            cj = s*DX+c*DY

            DX = int(round(ci))
            DY = int(round(cj))

            # get length of the current linked list
            size_boundary = list_bx.size()

            if size_boundary > 3:

              fx1 = list_bx[0]
              fx2 = list_bx[1]
              fy1 = list_by[0]
              fy2 = list_by[1]
              lx1 = list_bx[size_boundary-1]
              ly1 = list_by[size_boundary-1]
              lx2 = list_bx[size_boundary-2]
              ly2 = list_by[size_boundary-2]

              # check if the first and the last x and y are equal
              if (size_boundary > max_length) | \
              ((lx1 == fx2)&(lx2 == fx1)&(ly1 == fy2)&(ly2 == fy1)):
                  # remove the last element
                  list_bx.pop_back()
                  list_by.pop_back()
                  break

    return list_bx, list_by


def _isbf(long[:, :] mask, long[:, :] mask_90, long[:, :] mask_180, long[:, :] mask_270, int x_start, int y_start, float max_length):

    cdef long nrows = mask.shape[0]
    cdef long ncols = mask.shape[1]

    # initialize boundary vector
    cdef vector[long] list_bx
    cdef vector[long] list_by

    # set default direction
    cdef long DX = 1
    cdef long DY = 0

    # set the number of rows and cols for ISBF
    cdef long row_isbf = 3
    cdef long col_isbf = 2

    cdef double angle

    # set size of boundary points: the size of X is equal to the size of Y
    cdef long size_boundary

    cdef long[:, :] h
    cdef long a, b, x, y

    # initialize cX and cY which indicate directions for each ISBF
    cdef vector[int] cX
    cdef vector[int] cY

    # set sin and cos
    cdef double s, c, cx, cy, ci, cj

    cdef long fx1, fx2, fy1, fy2
    cdef long lx1, lx2, ly1, ly2
    cdef long lx3, lx4, ly3, ly4

    cdef long i, j, t

    with nogil:
        # push the first x and y points
        list_bx.push_back(x_start);
        list_by.push_back(y_start);

    while(True):

        h = np.zeros((row_isbf, col_isbf), dtype=int)

        with nogil:
            with cython.boundscheck(False):
                # initialize a and b which are indices of ISBF
                a = 0
                b = 0

                # get length of the current linked list
                size_boundary = list_bx.size()

                x = list_bx[size_boundary-1]
                y = list_by[size_boundary-1]

                if (DX == 1) & (DY == 0):
                    for i in range(ncols-x-2, ncols-x+1):
                        for j in range(y-1, y+1):
                            h[a, b] = mask_90[i, j]
                            b = b + 1
                        b = 0
                        a = a + 1
                    angle = M_PI/2

                elif (DX == 0) & (DY == -1):
                    for i in range(y-1, y+2):
                        for j in range(x-1, x+1):
                            h[a, b] = mask[i, j]
                            b = b + 1
                        b = 0
                        a = a + 1
                    angle = 0

                elif (DX == -1) & (DY == 0):
                    for i in range(x-1, x+2):
                        for j in range(nrows-y-2, nrows-y):
                            h[a, b] = mask_270[i, j]
                            b = b + 1
                        b = 0
                        a = a + 1
                    angle = 3*M_PI/2

                else:
                    for i in range(nrows-y-2, nrows-y+1):
                        for j in range(ncols-x-2, ncols-x):
                            h[a, b] = mask_180[i, j]
                            b = b + 1
                        b = 0
                        a = a + 1
                    angle = M_PI

                cX = vector[int](1)
                cY = vector[int](1)

                if h[1, 0] == 1:
                    # 'left' neighbor
                    cX[0] = -1
                    cY[0] = 0
                    DX = -1
                    DY = 0
                else:
                    if (h[2][0] == 1) & (h[2][1] != 1):
                        # inner-outer corner at left-rear
                        cX[0] = -1
                        cY[0] = 1
                        DX = 0
                        DY = 1
                    else:
                        if h[0, 0] == 1:
                            if h[0, 1] == 1:
                                # inner corner at front
                                cX[0] = 0
                                cY[0] = -1
                                cX.push_back(-1)
                                cY.push_back(0)
                                DX = 0
                                DY = -1
                            else:
                                # inner-outer corner at front-left
                                cX[0] = -1
                                cY[0] = -1
                                DX = 0
                                DY = -1
                        elif h[0, 1] == 1:
                            # front neighbor
                            cX[0] = 0
                            cY[0] = -1
                            DX = 1
                            DY = 0
                        else:
                            # outer corner
                            DX = 0
                            DY = 1

                # transform points by incoming directions and add to contours
                s = sin(angle)
                c = cos(angle)

                if (cX[0]!=0) | (cY[0]!=0):

                    for t in range(cX.size()):

                        cx = c*cX[t] - s*cY[t]
                        cy = s*cX[t] + c*cY[t]

                        with gil:
                            list_bx.push_back(list_bx.back()+int(round(cx)))
                            list_by.push_back(list_by.back()+int(round(cy)))

                ci = c*DX - s*DY
                cj = s*DX + c*DY

                DX = int(round(ci))
                DY = int(round(cj))

                # get length of the current linked list
                size_boundary = list_bx.size()

                if size_boundary > 3:

                    fx1 = list_bx[0]
                    fx2 = list_bx[1]
                    fy1 = list_by[0]
                    fy2 = list_by[1]
                    lx1 = list_bx[size_boundary-1]
                    ly1 = list_by[size_boundary-1]
                    lx2 = list_bx[size_boundary-2]
                    ly2 = list_by[size_boundary-2]
                    lx3 = list_bx[size_boundary-3]
                    ly3 = list_by[size_boundary-3]
                    lx4 = list_bx[size_boundary-4]
                    ly4 = list_by[size_boundary-4]

                    # check if the first and the last x and y are equal
                    if (size_boundary > max_length) | \
                            ((lx1 == fx2)&(lx2 == fx1)&(ly1 == fy2)&(ly2 == fy1)):
                        # remove the last element
                        list_bx.pop_back()
                        list_by.pop_back()
                        break
                    if cX.size() == 2:
                        if (lx2 == fx2)&(lx3 == fx1)&(ly2 == fy2)&(ly3 == fy1):
                            list_bx.pop_back()
                            list_by.pop_back()
                            list_bx.pop_back()
                            list_by.pop_back()
                            break
                    # detect cycle
                    if (lx1 == lx3)&(ly1 == ly3)&(lx2 == lx4)&(ly2 == ly4):
                        list_bx.pop_back()
                        list_by.pop_back()
                        list_bx.pop_back()
                        list_by.pop_back()
                        # change direction from M_PI to 3*M_PI/2
                        if (DX == 0) & (DY == 1):
                            DX = -1
                            DY = 0
                        # from M_PI/2 to M_PI
                        elif (DX == 1) & (DY == 0):
                            DX = 0
                            DY = 1
                        # from 0 to M_PI/2
                        elif (DX == 0) & (DY == -1):
                            DX = 1
                            DY = 0
                        else:
                            DX = 0
                            DY = -1

    return list_bx, list_by
