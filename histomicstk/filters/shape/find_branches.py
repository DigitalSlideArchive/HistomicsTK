import numpy as np
import scipy as sp


def find_branches(im_input):
    """
    Implements a branch detecting filter using using bifurcation patterns.

    Parameters
    ----------
    im_input : array_like
        M x N binary image. Objects to be detected are white.

    Returns
    -------
    Branches : array_like
        M x N binary image of branches.

    References
    ----------
    .. [1] M.A. Olsen et al "Convolution approach for feature detection in
           topological skeletons obtained from vascular patterns,"
           IEEE CIBIM, pp.163-167, 2011.
    """

    Branches = np.zeros_like(im_input)

    # set default kernels to detect branches
    kernel = []

    kernel.append(np.array([[0, 1, 0],
                            [1, 1, 1],
                            [0, 0, 0]]))

    kernel.append(np.array([[0, 1, 0],
                            [1, 1, 0],
                            [0, 0, 1]]))

    kernel.append(np.array([[0, 0, 1],
                            [0, 1, 0],
                            [1, 0, 1]]))

    kernel.append(np.array([[0, 0, 1],
                            [1, 1, 0],
                            [0, 0, 1]]))

    kernel.append(np.array([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]]))

    kernel.append(np.array([[1, 0, 1],
                            [0, 1, 0],
                            [1, 0, 1]]))

    # find branches
    for i in range(len(kernel)):
        current_kernel = kernel[i]
        if i < 4:
            for j in range(4):
                current_kernel = np.rot90(current_kernel)
                im_conv = sp.signal.convolve2d(im_input, current_kernel, boundary='fill', mode='same')
                Branches[np.where(im_conv > 3)] = 1
        else:
            im_conv = sp.signal.convolve2d(im_input, current_kernel, boundary='fill', mode='same')
            Branches[np.where(im_conv > 4)] = 1

    return Branches
