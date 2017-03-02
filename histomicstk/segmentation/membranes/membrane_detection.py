import histomicstk.filters.shape as htk_shape_filters

import numpy as np
import scipy as sp

import skimage.color
import skimage.filters
import skimage.io
import skimage.measure
import skimage.morphology


def membrane_detection(I, min_sigma=1, max_sigma=3, beta=4, c=2,
                       f_threshold=0.4, min_lsize=100,
                       b_dilation=1, b_split_dilation=3):
    """Performs membrane detection.
    Takes as input a dab-deconvolved image and performs the membrane filtering
    using eigenvalue analysis of scale-space blur image Hessians to detect
    membranes. This also detects membrane branches using bifurcation patterns.

    Parameters
    ----------
    I : array_like
        A DAB intensity image obtained from ColorDeconvolution.
    min_sigma : int
        Minimum vector of gaussian deviations used for membrane filter.
        Default value = 1.
    max_sigma : int
        Maximum vector of gaussian deviations used for membrane filter.
        Default value = 3.
    beta : double
        Scaling property of deviation from membrane. Decrease to increase
        importance of deviation in final score. Default value = 4.
    c : double
        Scaling property of quadratic curvature magnitude. Decrease to
        increase the importance of curvature magnitude in final score.
        Default value = 2.
    f_threshold : double
        A foreground threshold of membrane. Used with threshold_otsu.
        Default value = 0.4.
    min_lsize : int
        Minimum label size to be removed. Default value = 100.
    min_lsize : int
        Minimum label size to be removed. Default value = 100.
    b_dilation : int
        Branch dilation factor. Default value = 1.
    b_split_dilation : int
        Branch dilation factor to be splited. Default value = 3.

    Returns
    -------
    Label : array_like
        Label image where positive values correspond to membrane labels.
    Split : array_like
        Splitted image where positive values correspond to splitted points.
    Branches : array_like
        Branche image of where positive values correspond to membrane branches.

    References
    ----------
    .. [1] M.A. Olsen et al "Convolution approach for feature detection in
           topological skeletons obtained from vascular patterns,"
           IEEE CIBIM, pp.163-167, 2011.
    """

    # membrane filtering
    im_deviation, im_thetas = htk_shape_filters.membranefilter(
        I, np.arange(min_sigma, max_sigma+1),
        beta, c
    )
    im_deviation = 255 * im_deviation / im_deviation.max()

    # membrane thresholding
    im_mask = im_deviation > f_threshold * \
        skimage.filters.threshold_otsu(im_deviation)
    im_opened = sp.ndimage.binary_opening(
        im_mask, structure=np.ones((2, 2))
    ).astype(np.int)

    # skeletonization
    im_skeleton = skimage.morphology.skeletonize(im_opened)

    # default kernels to be used for detecting branches
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

    Branches = np.zeros_like(im_skeleton)

    # find branches
    for i in range(len(kernel)):
        current_kernel = kernel[i]
        if i < 4:
            for j in range(4):
                current_kernel = np.rot90(current_kernel)
                im_conv = sp.signal.convolve2d(
                    im_skeleton, current_kernel, boundary='fill', mode='same'
                )
                Branches[np.where(im_conv > 3)] = 1
        else:
            im_conv = sp.signal.convolve2d(
                im_skeleton, current_kernel, boundary='fill', mode='same'
            )
            Branches[np.where(im_conv > 4)] = 1

    # small labels removal
    im_split = im_skeleton & ~sp.ndimage.binary_opening(
        Branches,
        structure=skimage.morphology.disk(b_split_dilation)
    )

    # dilate regions
    im_split = sp.ndimage.binary_dilation(
        im_split,
        structure=skimage.morphology.disk(b_dilation)
    )

    # get label mask
    im_label, num_features = sp.ndimage.label(im_split)
    im_label_mask = np.zeros_like(im_label)

    # remove small labels
    for i in range(num_features):
        if len(np.where(im_label == i+1)[0]) > min_lsize:
            im_label_mask[np.where(im_label == i+1)] = 1

    # remove branches in the small labels
    Branches = Branches & im_label_mask



    # membrane label detection
    Split = im_label_mask & ~sp.ndimage.binary_dilation(
        Branches,
        structure=skimage.morphology.disk(b_split_dilation)
    )

    # dilate regions
    Split = sp.ndimage.binary_dilation(
        Split,
        structure=skimage.morphology.disk(b_dilation)
    )

    # get labels from the splited regions
    Label, num_features = sp.ndimage.label(Split)

    return Label, Split, Branches
