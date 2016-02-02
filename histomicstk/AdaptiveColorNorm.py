def AdaptiveColorNorm(I, Wstart, Wtarget):
    """Adaptive color normalization via sparse non-negative matrix factorization

    This method uses matrix factorization to adaptively identify a color
    deconvolution matrix that produces a sparse staining pattern for the
    input image `I`. A hot start initialization `Wstart` is used to improve
    convergence. The dynamic range of each stain for the input images is then
    stretched to match the range of the target image. Finally the input image
    is reconstructed with the color deconvolution stain matrix `Wtarget`.
    """
    pass
