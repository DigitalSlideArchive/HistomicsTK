import numpy as np

from histomicstk.segmentation import label


def min_model(I, Delta=0.3, MaxLength=255, Compaction=3,
              MinArea=100, MinWidth=5, MinDepth=2, MinConcavity=np.inf):
    """Performs a nuclear segmentation using a gradient contour tracing and
    geometry splitting algorithm. Implemented from the reference below.

    Parameters
    ----------
    I : array_like
        An intensity image used for analyzing local minima/maxima and
        gradients. Dimensions M x N.
    Delta : float
        Fractional difference threshold between minima/maxima pairs to
        be included in seed point detection. Fractional difference
        ([0, 1]) in total image range e.g. Delta = 0.3 with a uint8
        input would translate to 0.3 * 255. Default value = 0.3.
    MaxLength : int
        Maximum allowable contour length. Default value = 255.
    Compaction : int
        Factor used in compacting objects to remove thin spurs. Refered to as
        'd' in the paper. Default value = 3.
    MinArea : int
        Minimum area of objects to analyze. Default value = 100.
    MinWidth : int
        Minimum max-width of objects to analyze. Default value = 5.
    MinDepth : float
        Minimum depth of concavities to consider during geometric splitting.
        Default value = 2.
    MinConcavity : float
        Minimum concavity score to consider when performing for geometric
        splitting. Default value = np.inf.

    Notes
    -----
    Objects are assumed to be dark (as nuclei in hematoxylin channel from color
    deconvolution). Smoothing improves accuracy and computation time by
    eliminating spurious seed points. Specifying a value for 'Delta' prevents
    shallow transitions from being included, also reducing computation time and
    increasing specificity.

    Returns
    -------
    X : array_like
        A 1D array of horizontal coordinates of contour seed pixels for
        tracing.
    Y : array_like
        A 1D array of the vertical coordinates of seed pixels for tracing.
    Min : array_like
        A 1D array of the corresponding minimum values for contour tracing of
        seed point X, Y.
    Max : array_like
        A 1D array of the corresponding maximum values for contour tracing of
        seed point X, Y.

    See Also
    --------
    histomicstk.segmentation.label.trace_object_boundaries

    References
    ----------
    .. [#] S. Weinert et al "Detection and Segmentation of Cell Nuclei in
       Virtual Microscopy Images: A Minimum-Model Approach" in Nature
       Scientific Reports,vol.2,no.503, doi:10.1038/srep00503, 2012.

    """

    # identify contour seed points
    X, Y, Min, Max = seed_contours(I, Delta)

    # trace contours from seeds
    cXs, cYs = trace_contours(I, X, Y, Min, Max, MaxLength=255)

    # score successfully traced contours
    Scores = score_contours(I, cXs, cYs)

    # construct label image from scored contours
    Label = label_contour(I.shape, cXs, cYs, Scores)

    # compact contours to remove spurs - the paper calls this "optimization"
    Label = label.compact(Label, Compaction)

    # cleanup label image
    Label = label.split(Label)
    Label = label.area_open(Label, MinArea)
    Label = label.width_open(Label, MinWidth)

    # split objects with concavities
    Label = split_concavities(Label, MinDepth, MinConcavity)

    return Label


def seed_contours(I, Delta=0.3):  # noqa
    """Detects seed pixels for contour tracing by finding max-gradient points
    between local minima and maxima in an intensity image.

    Parameters
    ----------
    I : array_like
        An intensity image used for analyzing local minima/maxima and
        gradients. Dimensions M x N.
    Delta : float
        Fractional difference threshold between minima/maxima pairs to
        be included in seed point detection. Fractional difference
        ([0, 1]) in total image range e.g. Delta = 0.3 with a uint8
        input would translate to 0.3 * 255.  Default value = 0.3.

    Notes
    -----
    Objects are assumed to be dark (as nuclei in hematoxylin channel from color
    deconvolution). Smoothing improves accuracy and computation time by
    eliminating spurious seed points. Specifying a value for 'Delta' prevents
    shallow transitions from being included, also reducing computation time and
    increasing specificity.

    Returns
    -------
    X : array_like
        A 1D array of horizontal coordinates of contour seed pixels for
        tracing.
    Y : array_like
        A 1D array of the vertical coordinates of seed pixels for tracing.
    Min : array_like
        A 1D array of the corresponding minimum values for contour tracing of
        seed point X, Y.
    Max : array_like
        A 1D array of the corresponding maximum values for contour tracing of
        seed point X, Y.

    See Also
    --------
    TraceBounds, SeedContours, MinimumModel

    References
    ----------
    .. [#] S. Weinert et al "Detection and Segmentation of Cell Nuclei in
    Virtual Microscopy Images: A Minimum-Model Approach" in Nature Scientific
    Reports,vol.2,no.503, doi:10.1038/srep00503, 2012.

    """

    # initialize outputs
    X = []
    Y = []
    Min = []
    Max = []

    # for each image row
    for i in np.arange(I.shape[0]):

        # calculate gradient
        Gradient = np.hstack((np.nan, I[i, 2:] - I[i, 0:-2], np.nan))

        # identify local maxima and minimia of row 'i' of 'I'
        Maxima = ((I[i, 1:-1] >= I[i, 0:-2]) &
                  (I[i, 1:-1] > I[i, 2:])).nonzero()[0] + 1
        Minima = ((I[i, 1:-1] < I[i, 0:-2]) &
                  (I[i, 1:-1] <= I[i, 2:])).nonzero()[0] + 1

        # identify transitions - start of intervals of monotonic non-increase
        dI = np.sign(I[i, 1:].astype(float) - I[i, 0:-1].astype(float))
        dI = np.hstack((dI, dI[-1]))
        Transitions = np.nonzero(dI == 1)[0]
        Transitions = np.hstack((Transitions, I.shape[1]-1))

        # define min/max neighbor pairs
        MinPair = []
        MaxPair = []
        if(Minima.size > 0) & (Maxima.size > 0):

            # initialize initial positions of min/max & transition indices
            MinPos = 0
            MaxPos = 0
            TranPos = 0

            # iterate through maxima, identifying relevant minima for each
            while MaxPos < Maxima.size:
                Index = np.nonzero(Minima > Maxima[MaxPos])[0]
                if Index.size:  # minima found beyond current maxima

                    # get position of next minimum in array 'Minima'
                    MinPos = Index[0]

                    # increment transition point to beyond current maxima
                    while (TranPos < Transitions.size) & \
                            (Transitions[TranPos] <= Maxima[MaxPos]):
                        TranPos += 1

                    # add minima to current maxima until transition is reached
                    while Minima[MinPos] <= Transitions[TranPos]:
                        MinPair.append(Minima[MinPos])
                        MaxPair.append(Maxima[MaxPos])
                        MinPos += 1
                        if(MinPos == Minima.size):
                            break

                    # increment maxima
                    MaxPos += 1

                else:  # no minima beyond current maxima - quit
                    break

            # convert maxima/minima pairs to numpy arrays
            Maxima = np.asarray(MaxPair)
            Minima = np.asarray(MinPair)

            # remove pairs that do not have at least one pixel between them
            Close = ((Minima - Maxima) < 2).nonzero()
            Maxima = np.delete(Maxima, Close)
            Minima = np.delete(Minima, Close)

            # skip to next row if no min/max pairs exist after location filter
            if (Minima.size == 0) | (Maxima.size == 0):
                continue

            # remove pairs that do not have sufficient intensity transitions
            if(Delta is not None):
                if np.issubdtype(I.dtype, np.integer):
                    Range = Delta * 255.0
                elif np.issubdtype(I.dtype, float):
                    Range = Delta * 1.0
                Shallow = (I[i, Maxima] - I[i, Minima] < Range).nonzero()
                Maxima = np.delete(Maxima, Shallow)
                Minima = np.delete(Minima, Shallow)

            # skip to next row if no min/max pairs exist after intensity filter
            if (Minima.size == 0) | (Maxima.size == 0):
                continue

            # identify max gradient locations within paired maxima/minima
            MinGrad = np.zeros(Maxima.shape, dtype=int)
            for j in np.arange(Maxima.size):
                MinGrad[j] = np.argmin(Gradient[Maxima[j]+1:Minima[j]]) + \
                    Maxima[j]+1

            # capture min, max values and add to list with seed coordinates
            if(Maxima.size > 0):
                X.extend(MinGrad)
                Y.extend(i * np.ones(Maxima.size))
                Min.extend(I[i, Minima])
                Max.extend(I[i, MinGrad])

    # convert outputs from lists to numpy arrays
    X = np.array(X, dtype=np.uint)
    Y = np.array(Y, dtype=np.uint)
    Min = np.array(Min, dtype=I.dtype)
    Max = np.array(Max, dtype=I.dtype)

    # return seed pixels positions and intensity range intervals
    return X, Y, Min, Max


def trace_contours(I, X, Y, Min, Max, MaxLength=255):
    """Performs contour tracing of seed pixels in an intensity image using
    gradient information.

    Parameters
    ----------
    I : array_like
        An intensity image used for analyzing local minima/maxima and
        gradients. Dimensions M x N.
    X : array_like
        A 1D array of horizontal coordinates of contour seed pixels for
        tracing.
    Y : array_like
        A 1D array of the vertical coordinates of seed pixels for tracing.
    Min : array_like
        A 1D array of the corresponding minimum values for contour tracing of
        seed point X, Y.
    Max : array_like
        A 1D array of the corresponding maximum values for contour tracing of
        seed point X, Y.
    MaxLength : int
        Maximum allowable contour length. Default value = 255.

    Notes
    -----
    Can be computationally expensive for large numbers of contours. Use
    smoothing and delta thresholding when seeding contours to reduce burden.

    Returns
    -------
    cXs : list
        A list of 1D numpy arrays defining the horizontal coordinates of object
        boundaries.
    cYs : list
        A list of 1D numpy arrays defining the vertical coordinates of object
        boundaries.

    See Also
    --------
    SeedContours, ScoreContours, MinimumModel

    References
    ----------
    .. [#] S. Weinert et al "Detection and Segmentation of Cell Nuclei in
    Virtual Microscopy Images: A Minimum-Model Approach" in Nature Scientific
    Reports,vol.2,no.503, doi:10.1038/srep00503, 2012.

    """

    # initialize list of lists containing contours
    cXs = []
    cYs = []

    # process each seed pixel sequentially
    for i in np.arange(X.size):

        # capture window surrounding (X[i], Y[i])
        W = I[max(0, Y[i]-np.ceil(MaxLength/2.0)):
              min(I.shape[0]+1, Y[i]+np.ceil(MaxLength/2.0)+1),
              max(0, X[i]-np.ceil(MaxLength/2.0)):
              min(I.shape[1]+1, X[i]+np.ceil(MaxLength/2.0))+1]

        # binary threshold corresponding to seed pixel 'i'
        W = (W <= Max[i]) & (W >= Min[i])

        # embed with center pixel in middle of padded window
        Embed = np.zeros((W.shape[0]+2, W.shape[1]+2), dtype=np.bool)
        Embed[1:-1, 1:-1] = W

        # calculate location of (X[i], Y[i]) in 'Embed'
        pX = X[i] - max(0, X[i]-np.ceil(MaxLength/2.0)) + 1
        pY = Y[i] - max(0, Y[i]-np.ceil(MaxLength/2.0)) + 1

        # trace boundary, check stopping condition, append to list of contours
        cX, cY = label.trace_object_boundaries(Embed, conn=4,
                                               x_start=pX, y_start=pY,
                                               MaxLength=MaxLength)
        if cX[0][0] == cX[0][-1] and cY[0][0] == cY[0][-1] and\
                len(cX[0]) <= MaxLength:

            # add window offset to contour coordinates
            cX[0] = [
                x + max(0, X[i]-np.ceil(MaxLength/2.0)) - 1 for x in cX[0]
            ]
            cY[0] = [
                y + max(0, Y[i]-np.ceil(MaxLength/2.0)) - 1 for y in cY[0]
            ]

            # append to list of candidate contours
            cXs.append(np.array(cX[0], dtype=np.uint32))
            cYs.append(np.array(cY[0], dtype=np.uint32))

    return cXs, cYs


def score_contours(I, cXs, cYs):
    """Scores boundary contours using gradient information. Implemented from
    the reference below. Each contour is weighted by the average gradient and
    number of local gradient maxima along its path.

    Parameters
    ----------
    I : array_like
        An intensity image used for analyzing local minima/maxima and
        gradients. Dimensions M x N.
    cXs : list
        A list of 1D numpy arrays defining the horizontal coordinates of object
        boundaries.
    cYs : list
        A list of 1D numpy arrays defining the vertical coordinates of object
        boundaries.

    Notes
    -----
    Implemented from the reference below.

    Returns
    -------
    Scores : array_like
        A 1D array of horizontal coordinates of contour seed pixels for
        tracing.

    See Also
    --------
    TraceContours, LabelContour, MinimumModel

    References
    ----------
    .. [#] S. Weinert et al "Detection and Segmentation of Cell Nuclei in
    Virtual Microscopy Images: A Minimum-Model Approach" in Nature Scientific
    Reports,vol.2,no.503, doi:10.1038/srep00503, 2012.

    """
    import scipy.ndimage.filters as ft

    # initialize output
    Scores = np.zeros(len(cXs))

    # generate Sobel filter response from input intensity image 'I'
    Gradients = ft.sobel(I, mode='mirror')

    # generate local max in 3 x 3 window of Gradients
    Maxima = ft.maximum_filter(Gradients, size=3, mode='mirror')

    # generate score for each contour
    for i in np.arange(len(cXs)):

        # get gradient pixels, local max gradient pixels
        cG = Gradients[cYs[i], cXs[i]]
        cMax = Maxima[cYs[i], cXs[i]]

        # calculate mean gradient
        MG = np.sum(np.abs(Gradients[cYs[i], cXs[i]])) / len(cXs[i])

        # calculate gradient fit
        GF = np.sum(cG == cMax) / len(cXs[i])

        # compute score as product of mean gradient and gradient fit
        Scores[i] = MG * GF

    return Scores


def label_contour(Shape, cXs, cYs, Scores):
    """Constructs a label image from scored contours. Masks for contours with
    low priority/score are placed first into the label image and then are
    overwritten by higher priority contours.

    Parameters
    ----------
    Shape : tuple
        The shape tuple of the desired label image (height, width).
    cXs : list
        A list of 1D numpy arrays defining the horizontal coordinates of object
        boundaries.
    cYs : list
        A list of 1D numpy arrays defining the vertical coordinates of object
        boundaries.
    Scores : array_like
        A 1D array of horizontal coordinates of contour seed pixels for
        tracing.

    Notes
    -----
    Can produce a large number of thin "halo" objects surrouding the objects
    with higher scores. These can be removed by filtering object width in the
    resulting label image.

    Returns
    -------
    Label : array_like
        A uint32 label image.

    See Also
    --------
    ScoreContours, TraceContours, MinimumModel

    References
    ----------
    .. [#] S. Weinert et al "Detection and Segmentation of Cell Nuclei in
    Virtual Microscopy Images: A Minimum-Model Approach" in Nature Scientific
    Reports,vol.2,no.503, doi:10.1038/srep00503, 2012.

    """
    from skimage.draw import polygon

    # initialize label image
    Label = np.zeros(Shape, dtype=np.dtype('uint32'))

    # sort contours by scores
    Order = np.argsort(Scores)

    # loop over sorted contours, from least to most prominently scores
    for i in np.arange(len(cXs)):

        # get limits on contour
        xMin = np.min(cXs[Order[i]])
        xMax = np.max(cXs[Order[i]])
        yMin = np.min(cYs[Order[i]])
        yMax = np.max(cYs[Order[i]])

        # extract portion of existing label image
        T = Label[yMin:yMax+1, xMin:xMax+1]

        # generate mask for object 'Order[i]' from polygon
        Mask = polygon(cYs[Order[i]]-yMin, cXs[Order[i]]-xMin, T.shape)

        # replace non-zero areas with value 'i'
        T[Mask] = i

    return Label


def split_concavities(Label, MinDepth=4, MinConcavity=np.inf):  # noqa: C901
    """Performs splitting of objects in a label image using geometric scoring
    of concavities. Attempts to perform splits at narrow regions that are
    perpendicular to the object's convex hull boundaries.

    Parameters
    ----------
    Label : array_like
        A uint32 label image.
    MinDepth : float
        Minimum depth of concavities to consider during geometric splitting.
        Default value = 2.
    MinConcavity : float
        Minimum concavity score to consider when performing for geometric
        splitting. Default value = np.inf.

    Notes
    -----
    Can produce a large number of thin "halo" objects surrouding the objects
    with higher scores. These can be removed by filtering object width in the
    resulting label image.

    Returns
    -------
    Label : array_like
        A uint32 label image.

    See Also
    --------
    label_contours, min_model

    References
    ----------
    .. [#] S. Weinert et al "Detection and Segmentation of Cell Nuclei in
    Virtual Microscopy Images: A Minimum-Model Approach" in Nature Scientific
    Reports,vol.2,no.503, doi:10.1038/srep00503, 2012.

    """
    import scipy.ndimage.measurements as ms
    import scipy.ndimage.morphology as mp
    import skimage.morphology as mo

    # use shape profiles to split objects with concavities
    # copy input label image
    Convex = Label.copy()

    # condense label image
    if np.unique(Convex).size-1 != Convex.max():
        Convex = label.condense(Convex)

    # get locations of objects in initial image
    Locations = ms.find_objects(Convex)

    # initialize number of labeled objects and record initial count
    Total = Label.max()

    # initialize loop counter
    i = 1

    while i <= Total:

        # get object window from label image
        if i < len(Locations):
            W = Convex[Locations[i-1]]
        else:
            Locations = ms.find_objects(Convex)
            W = Convex[Locations[i-1]]

        # embed masked object in padded boolean array
        Mask = np.zeros((W.shape[0]+2, W.shape[1]+2), dtype=np.bool)
        Mask[1:-1, 1:-1] = W == i

        # generate convex hull of object
        Hull = mo.convex_hull_image(Mask)

        # generate boundary coordinates, trim duplicate point
        X, Y = label.trace_object_boundaries(Mask, conn=8)
        X = np.array(X[0][:-1], dtype=np.uint32)
        Y = np.array(Y[0][:-1], dtype=np.uint32)

        # calculate distance transform of object boundary pixels to convex hull
        Distance = mp.distance_transform_edt(Hull)
        D = Distance[Y, X] - 1

        # generate linear index of positions
        Linear = np.arange(X.size)

        # rotate boundary counter-clockwise until start position is on hull
        while(D[0] != 0):
            X = np.roll(X, -1)
            Y = np.roll(Y, -1)
            D = np.roll(D, -1)
            Linear = np.roll(Linear, -1)

        # find runs of concave pixels with length > 1
        Concave = (D > 0).astype(np.int)
        Start = np.where((Concave[1:] - Concave[0:-1]) == 1)[0]
        Stop = np.where((Concave[1:] - Concave[0:-1]) == -1)[0] + 1
        if(Stop.size == Start.size - 1):
            Stop = np.append(Stop, 0)

        # extract depth profiles, indices, distances for each run
        iX = []
        iY = []
        Depths = []
        Length = np.zeros(Start.size)
        MaxDepth = np.zeros(Start.size)
        for j in np.arange(Start.size):
            if(Start[j] < Stop[j]):
                iX.append(X[Start[j]:Stop[j]+1])
                iY.append(Y[Start[j]:Stop[j]+1])
                Depths.append(D[Start[j]:Stop[j]+1])
            else:  # run terminates at beginning of sequence
                iX.append(np.append(X[Start[j]:], X[0]))
                iY.append(np.append(Y[Start[j]:], Y[0]))
                Depths.append(np.append(D[Start[j]:], D[0]))
            Length[j] = iX[j].size
            MaxDepth[j] = np.max(Depths[j])

        # filter based on concave contour length and max depth
        Keep = np.where((Length > 1) & (MaxDepth >= MinDepth))[0]
        Start = Start[Keep]
        Stop = Stop[Keep]
        iX = [iX[Ind].astype(dtype=float) for Ind in Keep]
        iY = [iY[Ind].astype(dtype=float) for Ind in Keep]
        Depths = [Depths[Ind] for Ind in Keep]

        # attempt cutting if more than 1 sequence is found
        if Start.size > 1:

            # initialize containers to hold cut scores, optimal cut locations
            Scores = np.inf * np.ones((Start.size, Start.size))
            Xcut1 = np.zeros((Start.size, Start.size), dtype=np.uint32)
            Ycut1 = np.zeros((Start.size, Start.size), dtype=np.uint32)
            Xcut2 = np.zeros((Start.size, Start.size), dtype=np.uint32)
            Ycut2 = np.zeros((Start.size, Start.size), dtype=np.uint32)

            # compare candidates pairwise between all runs and score
            for j in np.arange(Start.size):

                # get list of 'j' candidates that pass depth threshold
                jCandidates = np.where(Depths[j] >= MinDepth)[0]

                for k in np.arange(j+1, Start.size):

                    # get list of 'k' candidates that pass depth threshold
                    kCandidates = np.where(Depths[k] >= MinDepth)[0]

                    # initialize minimum score and cut locations
                    minScore = np.inf
                    minj = -1
                    mink = -1

                    # loop over each coordinate pair for concavities j,k
                    for a in np.arange(jCandidates.size):
                        for b in np.arange(kCandidates.size):

                            # calculate length score
                            Ls = length_score(iX[j][jCandidates[a]],
                                              iY[j][jCandidates[a]],
                                              iX[k][kCandidates[b]],
                                              iY[k][kCandidates[b]],
                                              Depths[j][jCandidates[a]],
                                              Depths[k][kCandidates[b]])

                            # calculate angle score
                            As = angle_score(iX[j][0], iY[j][0],
                                             iX[j][-1], iY[j][-1],
                                             iX[k][0], iY[k][0],
                                             iX[k][-1], iY[k][-1],
                                             iX[j][jCandidates[a]],
                                             iY[j][jCandidates[a]],
                                             iX[k][kCandidates[b]],
                                             iY[k][kCandidates[b]])

                            # combine scores
                            Score = (Ls + As) / 2

                            # replace if improvement
                            if Score < minScore:
                                minScore = Score
                                Scores[j, k] = minScore
                                minj = jCandidates[a]
                                mink = kCandidates[b]

                    # record best cut location
                    Xcut1[j, k] = iX[j][minj]
                    Ycut1[j, k] = iY[j][minj]
                    Xcut2[j, k] = iX[k][mink]
                    Ycut2[j, k] = iY[k][mink]

            # pick the best scoring candidates and cut if needed
            ArgMin = np.unravel_index(Scores.argmin(), Scores.shape)
            if Scores[ArgMin[0], ArgMin[1]] <= MinConcavity:

                # perform cut
                SplitMask = cut(Mask,
                                Xcut1[ArgMin[0], ArgMin[1]].astype(float),
                                Ycut1[ArgMin[0], ArgMin[1]].astype(float),
                                Xcut2[ArgMin[0], ArgMin[1]].astype(float),
                                Ycut2[ArgMin[0], ArgMin[1]].astype(float))

                # re-label cut image
                SplitLabel = ms.label(SplitMask)[0]

                # increment object count, and label new object at end
                SplitLabel[SplitLabel > 1] = Total + 1
                Total += 1

                # label object '1' with current object value
                SplitLabel[SplitLabel == 1] = i

                # trim padding from corrected label image
                Mask = Mask[1:-1, 1:-1]
                SplitLabel = SplitLabel[1:-1, 1:-1]

                # update label image
                W[Mask] = SplitLabel[Mask]

            else:  # no cut made, move to next object
                i = i + 1
        else:  # no cuts to attempt, move to next object
            i = i + 1

    return Convex


def angle_score(ax1, ay1, bx1, by1, ax2, ay2, bx2, by2, cx1, cy1, cx2, cy2):
    """Scores the angles produced by cutting line (cx1, cy1)->(cx2, cy2) given
    the convex hull segments (ax1, ay1)->(bx1, by1) and (ax2, ay2)->(bx2, by2)
    spanning the concavities. See Figure 6 in reference below for a full
    illustration.

    Returns
    -------
    Score : float
        Angle score according to equation (6) from the reference.

    See Also
    --------
    SplitConcavities

    References
    ----------
    .. [#] S. Weinert et al "Detection and Segmentation of Cell Nuclei in
    Virtual Microscopy Images: A Minimum-Model Approach" in Nature Scientific
    Reports,vol.2,no.503, doi:10.1038/srep00503, 2012.

    """

    # calculate angle of hull at first concavity - y is inverted
    jHullAlpha = np.arctan2(ay1 - by1, bx1 - ax1)

    # calculate angle of cut at first concavity - y is inverted
    jCutAlpha = np.arctan2(cy2 - cy1, cx1 - cx2)

    # calculate angle of hull at second concavity - y is inverted
    kHullAlpha = np.arctan2(ay2 - by2, bx2 - ax2)

    # calculate angle of cut at second concavity - y is inverted
    kCutAlpha = np.arctan2(cy1 - cy2, cx2 - cx1)

    # calculate angle score
    Score = (np.abs(np.pi/2 - (jCutAlpha - jHullAlpha)) +
             np.abs(np.pi/2 - (kCutAlpha - kHullAlpha))) / np.pi

    return Score


def length_score(x1, y1, x2, y2, d1, d2):
    """Scores the length of the cutting line (x1, y1)->(x2, y2) made at a
    concavity depth of d1 and d2.

    Returns
    -------
    Score : float
        Angle score according to equation (5) from the reference.

    See Also
    --------
    SplitConcavities

    References
    ----------
    .. [#] S. Weinert et al "Detection and Segmentation of Cell Nuclei in
    Virtual Microscopy Images: A Minimum-Model Approach" in Nature Scientific
    Reports,vol.2,no.503, doi:10.1038/srep00503, 2012.

    """

    # calculate length of cut
    r = ((x1 - x2)**2 + (y1 - y2)**2) ** 0.5

    # normalize by total span across convex hull
    LengthScore = r / (r + d1 + d2)

    return LengthScore


def cut(Mask, x1, y1, x2, y2):
    """Performs a cut across a binary mask, zeroing pixels that round to
    positions on the line (x1, y1)->(x2, y2).

    Returns
    -------
    Cut : array_like
        A version of input Mask modified by cutting the line (x1, y1)->(x2, y2)

    See Also
    --------
    SplitConcavities

    References
    ----------
    .. [#] S. Weinert et al "Detection and Segmentation of Cell Nuclei in
    Virtual Microscopy Images: A Minimum-Model Approach" in Nature Scientific
    Reports,vol.2,no.503, doi:10.1038/srep00503, 2012.

    """

    # copy input
    Cut = Mask.copy()

    # calculate angle of line
    if(x1 < x2):
        theta = np.arctan2(y2-y1, x2-x1)
    else:
        theta = np.arctan2(y1-y2, x1-x2)

    # define line length
    length = ((x1-x2)**2 + (y1-y2)**2)**0.5

    # define points along x-axis
    x = np.arange(-1, length+1, 0.1)
    y = np.zeros(x.shape)

    # rotate
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    rotated = R.dot(np.vstack((x, y)))

    # translate
    if(x1 < x2):
        xr = rotated[0, :] + x1
        yr = rotated[1, :] + y1
    else:
        xr = rotated[0, :] + x2
        yr = rotated[1, :] + y2

    # take floor of coordinates
    xr = np.round(xr).astype(np.uint32)
    yr = np.round(yr).astype(np.uint32)

    # remove any pairs containing negative numbers to avoid access violations
    negative = (xr < 0) | (yr < 0)
    xr = np.delete(xr, negative)
    yr = np.delete(yr, negative)

    # zero out these coordinates
    Cut[yr, xr] = False

    return Cut
