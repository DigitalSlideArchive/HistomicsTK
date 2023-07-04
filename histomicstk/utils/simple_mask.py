import numpy as np


def simple_mask(im_rgb, bandwidth=2, bgnd_std=2.5, tissue_std=30,
                min_peak_width=10, max_peak_width=25,
                fraction=0.10, min_tissue_prob=0.05):
    """Performs segmentation of the foreground (tissue)
    Uses a simple two-component Gaussian mixture model to mask tissue areas
    from background in brightfield H&E images. Kernel-density estimation is
    used to create a smoothed image histogram, and then this histogram is
    analyzed to identify modes corresponding to tissue and background. The
    mode peaks are then analyzed to estimate their width, and a constrained
    optimization is performed to fit gaussians directly to the histogram
    (instead of using expectation-maximization directly on the data which
    is more prone to local minima effects). A maximum-likelihood threshold
    is then derived and used to mask the tissue area in a binarized image.

    Parameters
    ----------
    im_rgb : array_like
        An RGB image of type unsigned char.
    bandwidth : double, optional
        Bandwidth for kernel density estimation - used for smoothing the
        grayscale histogram. Default value = 2.
    bgnd_std : double, optional
        Standard deviation of background gaussian to be used if
        estimation fails. Default value = 2.5.
    tissue_std: double, optional
        Standard deviation of tissue gaussian to be used if estimation fails.
        Default value = 30.
    min_peak_width: double, optional
        Minimum peak width for finding peaks in KDE histogram. Used to
        initialize curve fitting process. Default value = 10.
    max_peak_width: double, optional
        Maximum peak width for finding peaks in KDE histogram. Used to
        initialize curve fitting process. Default value = 25.
    fraction: double, optional
        Fraction of pixels to sample for building foreground/background
        model. Default value = 0.10.
    min_tissue_prob : double, optional
        Minimum probability to qualify as tissue pixel. Default value = 0.05.

    Returns
    -------
    im_mask : array_like
        A binarized version of `I` where foreground (tissue) has value '1'.

    See Also
    --------
    histomicstk.utils.sample_pixels

    """
    from scipy import signal
    from scipy.optimize import fmin_slsqp
    from scipy.stats import norm
    from skimage import color
    from sklearn.neighbors import KernelDensity

    # convert image to grayscale, flatten and sample
    im_rgb = 255 * color.rgb2gray(im_rgb)
    im_rgb = im_rgb.astype(np.uint8)
    num_samples = int(fraction * im_rgb.size)
    sI = np.random.choice(im_rgb.flatten(), num_samples)[:, np.newaxis]

    # kernel-density smoothed histogram
    KDE = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(sI)
    xHist = np.linspace(0, 255, 256)[:, np.newaxis]
    yHist = np.exp(KDE.score_samples(xHist))[:, np.newaxis]
    yHist = yHist / sum(yHist)

    # flip smoothed y-histogram so that background mode is on the left side
    yHist = np.flipud(yHist)

    # identify initial mean parameters for gaussian mixture distribution
    # take highest peak among remaining peaks as background
    Peaks = signal.find_peaks_cwt(yHist.flatten(),
                                  np.arange(min_peak_width, max_peak_width))
    BGPeak = Peaks[0]
    if len(Peaks) > 1:
        TissuePeak = Peaks[yHist[Peaks[1:]].argmax() + 1]
    else:  # no peak found - take initial guess at 2/3 distance from origin
        TissuePeak = xHist[int(np.round(0.66 * xHist.size))].item()

    # analyze background peak to estimate variance parameter via FWHM
    BGScale = estimate_variance(xHist, yHist, BGPeak)
    if BGScale == -1:
        BGScale = bgnd_std

    # analyze tissue peak to estimate variance parameter via FWHM
    TissueScale = estimate_variance(xHist, yHist, TissuePeak)
    if TissueScale == -1:
        TissueScale = tissue_std

    # solve for mixing parameter
    Mix = yHist[BGPeak] * (BGScale * (2 * np.pi)**0.5)
    try:
        if len(Mix) == 1:
            Mix = Mix[0]
    except Exception:
        pass

    # flatten kernel-smoothed histogram and corresponding x values for
    # optimization
    xHist = xHist.flatten()
    yHist = yHist.flatten()

    # define gaussian mixture model
    def gaussian_mixture(x, mu1, mu2, sigma1, sigma2, p):
        rv1 = norm(loc=mu1, scale=sigma1)
        rv2 = norm(loc=mu2, scale=sigma2)
        return p * rv1.pdf(x) + (1 - p) * rv2.pdf(x)

    # define gaussian mixture model residuals
    def gaussian_residuals(Parameters, y, x):
        mu1, mu2, sigma1, sigma2, p = Parameters
        yhat = gaussian_mixture(x, mu1, mu2, sigma1, sigma2, p)
        return sum((y - yhat) ** 2)

    # fit Gaussian mixture model and unpack results
    Parameters = fmin_slsqp(gaussian_residuals,
                            [BGPeak, TissuePeak, BGScale, TissueScale, Mix],
                            args=(yHist, xHist),
                            bounds=[(0, 255), (0, 255),
                                    (np.spacing(1), 10),
                                    (np.spacing(1), 50), (0, 1)],
                            iprint=0)

    muBackground = Parameters[0]
    muTissue = Parameters[1]
    sigmaBackground = Parameters[2]
    sigmaTissue = Parameters[3]
    p = Parameters[4]

    # create mask based on Gaussian mixture model
    Background = norm(loc=muBackground, scale=sigmaBackground)
    Tissue = norm(loc=muTissue, scale=sigmaTissue)
    pBackground = p * Background.pdf(xHist)
    pTissue = (1 - p) * Tissue.pdf(xHist)

    # identify maximum likelihood threshold
    Difference = pTissue - pBackground
    Candidates = np.nonzero(Difference >= 0)[0]
    Filtered = np.nonzero(xHist[Candidates] > muBackground)
    ML = xHist[Candidates[Filtered[0]][0]]

    # identify limits for tissue model (MinProb, 1-MinProb)
    Endpoints = np.asarray(Tissue.interval(1 - min_tissue_prob / 2))

    # invert threshold and tissue mean
    ML = 255 - ML
    muTissue = 255 - muTissue
    Endpoints = np.sort(255 - Endpoints)

    # generate mask
    im_mask = (im_rgb <= ML) & (im_rgb >= Endpoints[0]) & \
              (im_rgb <= Endpoints[1])
    im_mask = im_mask.astype(np.uint8)

    return im_mask


def estimate_variance(x, y, peak):
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
    peak : double
        index of peak in y to estimate variance of
    Returns
    -------
    scale : double
        Standard deviation of normal distribution approximating peak. Value is
        -1 if fitting process fails.
    See Also
    --------
    SimpleMask

    """

    # analyze peak to estimate variance parameter via FWHM
    peak = int(peak)
    Left = peak
    while y[Left] > y[peak] / 2 and Left >= 0:
        Left -= 1
        if Left == -1:
            break
    Right = peak
    while y[Right] > y[peak] / 2 and Right < y.size:
        Right += 1
        if Right == y.size:
            break
    if Left != -1 and Right != y.size:
        LeftSlope = y[Left + 1] - y[Left] / (x[Left + 1] - x[Left])
        Left = (y[peak] / 2 - y[Left]) / LeftSlope + x[Left]
        RightSlope = y[Right] - y[Right - 1] / (x[Right] - x[Right - 1])
        Right = (y[peak] / 2 - y[Right]) / RightSlope + x[Right]
        scale = (Right - Left) / 2.355
    if Left == -1:
        if Right == y.size:
            scale = -1
        else:
            RightSlope = y[Right] - y[Right - 1] / (x[Right] - x[Right - 1])
            Right = (y[peak] / 2 - y[Right]) / RightSlope + x[Right]
            scale = 2 * (Right - x[peak]) / 2.355
    if Right == y.size:
        if Left == -1:
            scale = -1
        else:
            LeftSlope = y[Left + 1] - y[Left] / (x[Left + 1] - x[Left])
            Left = (y[peak] / 2 - y[Left]) / LeftSlope + x[Left]
            scale = 2 * (x[peak] - Left) / 2.355
    try:
        if len(scale) == 1:
            scale = scale[0]
    except Exception:
        pass
    return scale
