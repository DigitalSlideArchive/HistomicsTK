

def EstimateVariance(x, y, Peak):
    """Estimates variance of a peak in a histogram using the FWHM of an
    approximate normal distribution.

    Starting from a user-supplied peak and histogram, this method traces down
    each side of the peak to estimate the full-width-half-maximum (FWHM) and
    variance of the peak. If tracing fails on either side, the FWHM is
    estimated as twice the HWHM.

    Parameters
    ----------
    x : array_like
        vector of x-histogram locations.
    y : array_like
        vector of y-histogram locations.
    Peak : double
        index of peak in y to estimate variance of

    Returns
    -------
    Scale : double
        Standard deviation of normal distribution approximating peak. Value is
        -1 if fitting process fails.

    See Also
    --------
    SimpleMask
    """

    # analyze peak to estimate variance parameter via FWHM
    Left = Peak
    while y[Left] > y[Peak] / 2 and Left >= 0:
        Left -= 1
        if Left == -1:
            break
    Right = Peak
    while y[Right] > y[Peak] / 2 and Right < y.size:
        Right += 1
        if Right == y.size:
            break
    if Left != -1 and Right != y.size:
        LeftSlope = y[Left + 1] - y[Left] / (x[Left + 1] - x[Left])
        Left = (y[Peak] / 2 - y[Left]) / LeftSlope + x[Left]
        RightSlope = y[Right] - y[Right - 1] / (x[Right] - x[Right - 1])
        Right = (y[Peak] / 2 - y[Right]) / RightSlope + x[Right]
        Scale = (Right - Left) / 2.355
    if Left == -1:
        if Right == y.size:
            Scale = -1
        else:
            RightSlope = y[Right] - y[Right - 1] / (x[Right] - x[Right - 1])
            Right = (y[Peak] / 2 - y[Right]) / RightSlope + x[Right]
            Scale = 2 * (Right - x[Peak]) / 2.355
    if Right == y.size:
        if Left == -1:
            Scale = -1
        else:
            LeftSlope = y[Left + 1] - y[Left] / (x[Left + 1] - x[Left])
            Left = (y[Peak] / 2 - y[Left]) / LeftSlope + x[Left]
            Scale = 2 * (x[Peak] - Left) / 2.355

    return Scale
