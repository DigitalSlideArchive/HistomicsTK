def sda_to_rgb(im_sda, I_0):
    """Transform input SDA image or matrix `im_sda` into RGB space.  This
    is the inverse of `rgb_to_sda` with respect to the first parameter

    Parameters
    ----------
    im_sda : array_like
        Image (MxNx3) or matrix (3xN) of pixels

    I_0 : float or array_like
        Background intensity, either per-channel or for all channels

    Note
    ----
    For compatibility purposes, passing I_0=None invokes the behavior of
    od_to_rgb.

    See Also
    --------
    histomicstk.preprocessing.color_conversion.rgb_to_sda,
    histomicstk.preprocessing.color_conversion.od_to_rgb,
    histomicstk.preprocessing.color_deconvolution.color_deconvolution,
    histomicstk.preprocessing.color_deconvolution.color_convolution

    """
    is_matrix = im_sda.ndim == 2
    if is_matrix:
        im_sda = im_sda.T

    od = I_0 is None
    if od:  # od_to_rgb compatibility
        I_0 = 256

    im_rgb = I_0 ** (1 - im_sda / 255.)
    return (im_rgb.T if is_matrix else im_rgb) - od
