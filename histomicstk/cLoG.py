import numpy as np
import scipy as sp


def cLoG(I, Mask, SigmaMin=30*1.414, SigmaMax=50*1.414):
    """Constrainted Laplacian of Gaussian filter.

    Takes as input a grayscale nuclear image and binary mask of cell nuclei,
    and uses the distance transform of the nuclear mask to constrain the LoG
    filter response of the image for nuclear seeding. Returns a LoG filter
    image of type float. Local maxima are used for seeding cells.

    Parameters
    ----------
    I : array_like
        A hematoxylin intensity image obtained from ColorDeconvolution. Objects
        are assumed to be dark with a light background.
    Mask : array_like
        A binary image where nuclei pixels have value 1/True, and non-nuclear
        pixels have value 0/False.
    SigmaMin : float
        A scalar defining the minimum scaled nuclear radius. Radius is scaled
        by sqrt(2). Default value = 30 * 2 ** 0.5.
    SigmaMax : float
        A scalar defining the maximum scaled nuclear radius. Radius is scaled
        by sqrt(2). Default value = 50 * 2 ** 0.5.

    Returns
    -------
    Iout : array_like
        A color image of type unsigned char where boundary pixels take
        on the color defined by the RGB-triplet 'Color'.

    See Also
    --------
    MergeSeeds

    References
    ----------
    .. [1] Y. Al-Kofahi et al "Improved Automatic Detection and Segmentation
    of Cell Nuclei in Histopathology Images" in IEEE Transactions on Biomedical
    Engineering,vol.57,no.4,pp.847-52, 2010.
    """

    # convert intensity image type to float if needed
    if I.dtype == np.uint8:
        I = I.astype(np.float)

    # generate distance map
    Distance = sp.ndimage.morphology.distance_transform_edt(Mask)

    # initialize constraint
    Constraint = np.maximum(SigmaMin, np.minimum(SigmaMax, 2*Distance))

    # initialize log filter response array
    Iout = np.finfo(Distance.dtype).min * np.ones(Mask.shape)

    # LoG filter over scales
    Start = np.floor(SigmaMin)
    Stop = np.ceil(SigmaMax)
    Sigmas = np.linspace(Start, Stop, Stop-Start+1)
    for Sigma in Sigmas:

        # generate normalized filter response
        Response = Sigma ** 2 * \
            sp.ndimage.filters.gaussian_laplace(I, Sigma, mode='constant',
                                                cval=0.0)

        # constrain response
        Map = Sigma < Constraint
        Response[~Map] = np.finfo(Distance.dtype).min

        # replace with maxima
        Iout = np.maximum(Iout, Response)

    # translate filtered image

    # replace min floats
    Iout[Iout == np.finfo(Distance.dtype).min] = 0

    return Iout
