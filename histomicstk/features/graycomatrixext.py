import numpy as np


def graycomatrixext(im_input, im_roi_mask=None,
                    offsets=None, num_levels=8, gray_limits=[0, 255],
                    symmetric=False, normed=True, exclude_boundary=False):
    """Computes gray-level co-occurence matrix.

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

        AngleXY  |  OFFSET
        -------  |  ------
        0        |  [0 D]
        45       |  [-D D]
        90       |  [-D 0]
        135      |  [-D -D]

        Default
            [1] for 1D,
            [[1, 0], [0, 1]] for 2D,
            [[1, 0, 0], [0, 1, 0], [0, 0, 1] for 3D and so on

    num_levels : unsigned int, optional
        An integer specifying the number of gray levels For example, if
        `NumLevels` is 8,  the intensity values of the input image are
        scaled so they are integers between 1 and 8.  The number of gray
        levels determines the size of the gray-level co-occurrence matrix.

        Default: 8 for numeric image, 2 for binary/logical image

    gray_limits : array_like, optional
        A two-element array specifying the desired input intensity range.
        Intensity values in the input image will be clipped into this range.

        Default: [0, 1] for binary/logical image, [0, 255] for numeric image

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
    ..  [1] Haralick, R.M., K. Shanmugan, and I. Dinstein, "Textural Features
        for Image Classification", IEEE Transactions on Systems, Man, and
        Cybernetics, Vol. SMC-3, 1973, pp. 610-621.

    ..  [2] Haralick, R.M., and L.G. Shapiro. Computer and Robot Vision:
        Vol. 1, Addison-Wesley, 1992, p. 459.
    """

    num_dims = len(im_input.shape)

    # roi mask
    if im_roi_mask is None:

        # compute glcm for whole input image
        im_roi_mask = np.ones_like(im_input, dtype='bool')

    else:

        # check sanity
        if im_input.shape != im_roi_mask.shape:
            raise ValueError('size mismatch between input image and roi mask')

    # offsets
    if offsets is None:

        # set default offset value
        offsets = np.identity(num_dims)

    else:

        # check sanity
        if offsets.shape[1] != num_dims:
            raise ValueError(
                'Dimension mismatch between input image and offsets'
            )

    num_offsets = offsets.shape[0]

    # scale input intensity image
    im_input -= gray_limits[0]
    im_input /= np.float(gray_limits[1] - gray_limits[0])
    im_input *= (num_levels - 1)
    im_input = np.round(im_input)

    # compute glcm for each offset
    glcm = np.zeros((num_levels, num_levels, num_offsets))

    im_input_flat = np.ravel(im_input)
    im_roi_mask_flat = np.ravel(im_roi_mask)

    roi_coord_ind = np.nonzero(im_roi_mask)
    roi_lin_ind = np.ravel_multi_index(roi_coord_ind, im_roi_mask.shape)

    for i in range(num_offsets):

        # compute indices of neighboring pixels by applying the offset
        neigh_coord_ind = roi_coord_ind

        for j in range(num_dims):
            neigh_coord_ind[j] += offsets[i, j]

        # throw out pixels with invalid neighbors
        neigh_valid = np.ones_like(neigh_coord_ind[0], dtype='bool')

        for j in range(num_dims):

            neigh_valid[neigh_coord_ind[j] < 0] = False
            neigh_valid[neigh_coord_ind[j] >= num_levels] = False

        for j in range(num_dims):
            neigh_coord_ind[j] = np.compress(neigh_valid,
                                             neigh_coord_ind[j], axis=0)

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
        cur_glcm = np.ravel(glcm[:, :, i])
        cur_glcm[pind] = pcount

        # symmetricize if asked for
        if symmetric:
            glcm[:, :, i] += glcm[:, :, i].T

        # normalize if asked
        if normed:
            glcm[:, :, i] /= glcm[:, :, i].sum()

    return glcm