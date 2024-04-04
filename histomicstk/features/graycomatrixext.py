import numpy as np


def graycomatrixext(im_input, im_roi_mask=None,
                    offsets=None, num_levels=None, gray_limits=None,
                    symmetric=False, normed=False, exclude_boundary=False):
    """Computes gray-level co-occurence matrix (GLCM) within a region of
    interest (ROI) of an image. GLCM is a 2D histogram/matrix containing the
    counts/probabilities of co-occuring intensity values at a given offset
    within an ROI of an image.

    Read the documentation to know the default values used for each of the
    optional parameter in different scenarios.

    Parameters
    ----------
    im_input : array_like
        Input single channel intensity image

    im_roi_mask : array_like, optional
        A binary mask specifying the region of interest within which
        to compute the GLCM. If not specified GLCM is computed for the
        the entire image.

        Default: None

    offsets : array_like, optional
        A (num_offsets, num_image_dims) array of offset vectors
        specifying the distance between the pixel-of-interest and
        its neighbor. Note that the first dimension corresponds to
        the rows.

        Because this offset is often expressed as an angle, the
        following table lists the offset values that specify common
        angles for a 2D image, given the pixel distance D.

        ===========  =============
        Angle (deg)  offset [y, x]
        ===========  =============
        0            [0 D]
        45           [-D D]
        90           [-D 0]
        135          [-D -D]
        ===========  =============

        Default
        - 1D: np.array([1])
        - 2D : numpy.array([ [1, 0], [0, 1], [1, 1], [1, -1] ])
        - 3D and higher: numpy.identity(num_image_dims)

    num_levels : unsigned int, optional
        An integer specifying the number of gray levels For example, if
        `NumLevels` is 8,  the intensity values of the input image are
        scaled so they are integers between 1 and 8.  The number of gray
        levels determines the size of the gray-level co-occurrence matrix.

        Default: 2 for binary/logical image, 32 for numeric image

    gray_limits : array_like, optional
        A two-element array specifying the desired input intensity range.
        Intensity values in the input image will be clipped into this range.

        Default: [0, 1] for boolean-valued image, [0, 255] for integer-valued
        image, and [0.0, 1.0] for-real valued image

    symmetric : bool, optional
        A boolean value that specifies whether or not the ordering of values
        in pixel pairs is considered while creating the GLCM matrix.

        For example, if `Symmetric` is True, then while calculating the
        number of times the value 1 is adjacent to the value 2, both
        1,2 and 2,1 pairings are counted. GLCM created in this way is
        symmetric across its diagonal.

        Default: False

    normed : bool, optional
        A boolean value specifying whether or not to normalize glcm.

        Default: False

    exclude_boundary : bool, optional
        Specifies whether or not to exclude a pixel-pair if the
        neighboring pixel in the pair is outside `im_roi_mask`.
        Has an effect only when `im_roi_mask` is specified.

        Default: False

    Returns
    -------
    glcm : array_like
        num_levels x num_levels x num_offsets array containing the GLCM
        for each offset.

    References
    ----------
    .. [#] Haralick, R.M., K. Shanmugan, and I. Dinstein, "Textural Features
       for Image Classification", IEEE Transactions on Systems, Man, and
       Cybernetics, Vol. SMC-3, 1973, pp. 610-621.

    .. [#] Haralick, R.M., and L.G. Shapiro. Computer and Robot Vision:
       Vol. 1, Addison-Wesley, 1992, p. 459.

    """
    num_dims = len(im_input.shape)

    # roi mask
    if im_roi_mask is None:

        # compute glcm for whole input image
        im_roi_mask = np.ones_like(im_input, dtype='bool')

    if im_input.shape != im_roi_mask.shape:
        msg = 'size mismatch between input image and roi mask'
        raise ValueError(msg)

    # gray_limits
    if gray_limits is None:

        gray_limits = _default_gray_limits(im_input)

    assert len(gray_limits) == 2
    assert gray_limits[0] < gray_limits[1]

    # num_levels
    if num_levels is None:

        num_levels = _default_num_levels(im_input)

    # offsets
    if offsets is None:

        # set default offset value
        offsets = _default_offsets(im_input)

    else:

        # check sanity
        if offsets.shape[1] != num_dims:
            msg = 'Dimension mismatch between input image and offsets'
            raise ValueError(
                msg,
            )

    num_offsets = offsets.shape[0]

    # scale input intensity image
    im_input = im_input.astype('float')
    im_input -= gray_limits[0]
    im_input /= float(gray_limits[1] - gray_limits[0])
    im_input *= (num_levels - 1)
    im_input = np.round(im_input).astype('int')

    # compute glcm for each offset
    glcm = np.zeros((num_levels, num_levels, num_offsets))

    im_input_flat = np.ravel(im_input)

    im_roi_mask_flat = np.ravel(im_roi_mask)

    roi_coord_ind = np.nonzero(im_roi_mask)

    roi_lin_ind = np.ravel_multi_index(roi_coord_ind, im_roi_mask.shape)

    for i in range(num_offsets):

        # compute indices of neighboring pixels by applying the offset
        neigh_coord_ind = [None] * len(roi_coord_ind)

        for j in range(num_dims):
            neigh_coord_ind[j] = roi_coord_ind[j] + offsets[i, j]

        # throw out pixels with invalid neighbors
        neigh_valid = np.ones_like(neigh_coord_ind[0], dtype='bool')

        for j in range(num_dims):

            neigh_valid[neigh_coord_ind[j] < 0] = False
            neigh_valid[neigh_coord_ind[j] >= im_roi_mask.shape[j]] = False

        for j in range(num_dims):
            neigh_coord_ind[j] = np.compress(neigh_valid, neigh_coord_ind[j],
                                             axis=0).astype(np.int64)

        neigh_lin_ind = np.ravel_multi_index(neigh_coord_ind,
                                             im_roi_mask.shape)

        if exclude_boundary:
            neigh_valid[im_roi_mask_flat[neigh_lin_ind] == 0] = False
            neigh_lin_ind = np.compress(neigh_valid, neigh_lin_ind, axis=0)

        valid_roi_lin_ind = np.compress(neigh_valid, roi_lin_ind, axis=0)

        # get intensities of pixel pairs which become coord indices in glcm
        p1 = np.take(im_input_flat, valid_roi_lin_ind, axis=0)
        p2 = np.take(im_input_flat, neigh_lin_ind, axis=0)

        # convert pixel-pair values to linear indices in glcm
        pind = np.ravel_multi_index((p1, p2), glcm.shape[:2])

        # find unique linear indices and their counts
        pind, pcount = np.unique(pind, return_counts=True)

        # put count of each linear index in glcm
        cur_glcm = np.zeros((num_levels, num_levels))
        cur_glcm_flat = np.ravel(cur_glcm)
        cur_glcm_flat[pind] = pcount

        # symmetricize if asked for
        if symmetric:
            cur_glcm += cur_glcm.T

        # normalize if asked
        if normed and cur_glcm.sum():
            cur_glcm /= cur_glcm.sum()

        glcm[:, :, i] = cur_glcm

    return glcm


def _default_gray_limits(im_input):

    assert isinstance(im_input, np.ndarray)

    if np.issubdtype(im_input.dtype, np.bool_):

        gray_limits = [0, 1]

    elif np.issubdtype(im_input.dtype, np.integer):

        gray_limits = [0, 255]

    elif np.issubdtype(im_input.dtype, np.floating):

        gray_limits = [0.0, 1.0]

    else:

        msg = 'The type of the argument im_input is invalid'
        raise ValueError(msg)

    return gray_limits


def _default_num_levels(im_input):

    assert isinstance(im_input, np.ndarray)

    if np.issubdtype(im_input.dtype, np.bool_):

        num_levels = 2

    elif np.issubdtype(im_input.dtype, np.number):

        num_levels = 32

    else:

        msg = 'The type of the argument im_input is invalid'
        raise ValueError(msg)

    return num_levels


def _default_offsets(im_input):

    num_dims = len(im_input.shape)

    if num_dims == 2:

        offsets = np.array([
            [0, 1], [1, 0], [1, 1], [1, -1],
        ])

    else:

        # TODO: need to come up with a better strategy for 3D and higher
        offsets = np.identity(num_dims)

    return offsets
