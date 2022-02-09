import numpy as np

from histomicstk.utils import gradient_diffusion


def gvf_tracking(I, Mask, K=1000, Diffusions=10, Mu=5, Lambda=5, Iterations=10,
                 dT=0.05):
    """
    Performs gradient-field tracking to segment smoothed images of cell nuclei.

    Takes as input a smoothed intensity or Laplacian-of-Gaussian filtered image
    and a foreground mask, and groups pixels by tracking them to mutual
    gradient sinks. Typically requires merging of sinks (seeds) as a post
    processing steps.

    Parameters
    ----------
    I : array_like
        Smoothed intensity or log-filtered response where nuclei regions have
        larger intensity values than background.
    Mask : array_like
        Binary mask where foreground objects have value 1, and background
        objects have value 0. Used to restrict influence of background vectors
        on diffusion process and to reduce tracking computations.
    K : float
        Number of steps to check for tracking cycle. Default value = 1000.
    Mu : float
        Weight parmeter from Navier-Stokes diffusion - weights divergence and
        Laplacian terms. Default value = 5.
    Lambda : float
        Weight parameter from Navier-Stokes diffusion - used to weight
        divergence. Default value = 5.
    Iterations : float
        Number of time-steps to use in Navier-Stokes diffusion. Default value =
        10.
    dT : float
        Timestep to be used in Navier-Stokes diffusion. Default value = 0.05.

    Returns
    -------
    Segmentation : array_like
        Label image where positive values correspond to foreground pixels that
        share mutual sinks.
    Sinks : array_like
        N x 2 array containing the (x,y) locations of the tracking sinks. Each
        row is an (x,y) pair - in that order.

    See Also
    --------
    histomicstk.utils.gradient_diffusion,
    histomicstk.segmentation.label.shuffle

    References
    ----------
    .. [#] G. Li et al "3D cell nuclei segmentation based on gradient flow
       tracking" in BMC Cell Biology,vol.40,no.8, 2007.

    """
    # get image shape
    M = I.shape[0]
    N = I.shape[1]

    # calculate gradient
    dy, dx = np.gradient(I)

    # diffusion iterations
    if Diffusions > 0:
        dx, dy = gradient_diffusion(dx, dy, Mask, Mu, Lambda, Diffusions,
                                    dT)

    # normalize to unit magnitude
    Mag = ((dx**2 + dy**2)**0.5 + np.finfo(float).eps)
    dy = dy / Mag
    dx = dx / Mag

    # define mask to track pixels that are mapped to a sink
    Mapped = np.zeros(I.shape)

    # define label image
    Segmentation = np.zeros(I.shape)

    # initialize lists of sinks
    Sinks = []

    # define coordinates for foreground pixels (Mask == 1)
    i, j = np.nonzero(Mask)

    # track pixels
    for index, (x, y) in enumerate(zip(j, i)):

        # initialize angle, trajectory length, novel flag, and allocation count
        phi = 0
        points = 1
        novel = 1
        alloc = 1

        # initialize trajectory
        Trajectory = np.zeros((K, 2))
        Trajectory[0, 0] = x
        Trajectory[0, 1] = y

        # track while angle defined by successive steps is < np.pi / 2
        while(phi < np.pi / 2):

            # calculate step
            xStep = round_float(dx[Trajectory[points-1, 1],
                                Trajectory[points-1, 0]])
            yStep = round_float(dy[Trajectory[points-1, 1],
                                Trajectory[points-1, 0]])

            # check image edge
            if ((Trajectory[points-1, 0] + xStep < 0) or
                    (Trajectory[points-1, 0] + xStep > N-1) or
                    (Trajectory[points-1, 1] + yStep < 0) or
                    (Trajectory[points-1, 1] + yStep > M-1)):
                break

            # add new point to trajectory list
            if points < K:  # buffer is not overrun
                Trajectory[points, 0] = Trajectory[points-1, 0] + xStep
                Trajectory[points, 1] = Trajectory[points-1, 1] + yStep

            else:  # buffer overrun

                # check for cycle
                cycle = detect_cycle(Trajectory, points)

                if cycle == points:  # no cycle, simple overflow. grow buffer.

                    # copy and reallocate
                    temp = Trajectory
                    Trajectory = np.zeros((K*alloc, 2))
                    Trajectory[K*(alloc-1):K*alloc, ] = temp
                    alloc += 1

                    # add new point
                    Trajectory[points, 0] = Trajectory[points-1, 0] + xStep
                    Trajectory[points, 1] = Trajectory[points-1, 1] + yStep

                else:  # overflow due to cycle, terminate tracking
                    points = cycle

            # check mapping
            if Mapped[Trajectory[points, 1], Trajectory[points, 0]] == 1:
                novel = 0
                phi = np.pi
            elif Mask[Trajectory[points, 1], Trajectory[points, 0]] == 0:
                phi = np.pi
            else:
                phi = np.arccos(dy[Trajectory[points-1, 1],
                                   Trajectory[points-1, 0]] *
                                dy[Trajectory[points, 1],
                                   Trajectory[points, 0]] +
                                dx[Trajectory[points-1, 1],
                                   Trajectory[points-1, 0]] *
                                dx[Trajectory[points, 1],
                                   Trajectory[points, 0]])

            # increment trajectory length counter
            points += 1

        # determine if sink is novel
        if novel == 1:

            # record sinks
            Sinks.append(Trajectory[points-1, ])

            # add trajectory to label image with new sink value, add mapping
            for j in range(points):
                Segmentation[Trajectory[j, 1], Trajectory[j, 0]] = len(Sinks)
                Mapped[Trajectory[j, 1], Trajectory[j, 0]] = 1

        else:

            # add trajectory to label image with sink value of final point
            for j in range(points):
                Segmentation[Trajectory[j, 1], Trajectory[j, 0]] = \
                    Segmentation[Trajectory[points-1, 1],
                                 Trajectory[points-1, 0]]

    # convert Sinks to numpy array
    Sinks = np.asarray(Sinks)

    return Segmentation, Sinks


def merge_sinks(Label, Sinks, Radius=5):
    """
    Merges attraction basins obtained from gradient flow tracking using
    sink locations.

    Parameters
    ----------
    Segmentation : array_like
        Label image where positive values correspond to foreground pixels that
        share mutual sinks.
    Sinks : array_like
        N x 2 array containing the (x,y) locations of the tracking sinks. Each
        row is an (x,y) pair - in that order.
    Radius : float
        Radius used to merge sinks. Sinks closer than this radius to one
        another will have their regions of attraction merged.
        Default value = 5.

    Returns
    -------
    Merged : array_like
        Label image where attraction regions are merged.

    """
    import skimage.morphology as mp
    from skimage import measure as ms

    # build seed image
    SeedImage = np.zeros(Label.shape)
    for i in range(Sinks.shape[0]):
        SeedImage[Sinks[i, 1], Sinks[i, 0]] = i+1

    # dilate sink image
    Dilated = mp.binary_dilation(SeedImage, mp.disk(Radius))

    # generate new labels for merged seeds, define memberships
    Labels = ms.label(Dilated)
    New = Labels[Sinks[:, 1].astype(np.int), Sinks[:, 0].astype(np.int)]

    # get unique list of seed clusters
    Unique = np.arange(1, New.max()+1)

    # generate new seed list
    Merged = np.zeros(Label.shape)

    # get pixel list for each sink object
    Props = ms.regionprops(Label.astype(np.int))

    # fill in new values
    for i in Unique:
        Indices = np.nonzero(New == i)[0]
        for j in Indices:
            Coords = Props[j].coords
            Merged[Coords[:, 0], Coords[:, 1]] = i

    return Merged


def detect_cycle(Trajectory, points):

    # initialize trajectory length
    length = 0

    # identify trajectory bounding box
    xMin = np.min(Trajectory[0:points, 0])
    xMax = np.max(Trajectory[0:points, 0])
    xRange = xMax - xMin + 1
    yMin = np.min(Trajectory[0:points, 1])
    yMax = np.max(Trajectory[0:points, 1])
    yRange = yMax - yMin + 1

    # fill in trajectory map
    Map = np.zeros((yRange, xRange))
    for i in range(points):
        if Map[Trajectory[i, 1]-yMin, Trajectory[i, 0]-xMin] == 1:
            break
        else:
            Map[Trajectory[i, 1]-yMin, Trajectory[i, 0]-xMin] = 1
        length += 1

    return length


def round_float(x):
    if x >= 0.0:
        t = np.ceil(x)
        if t - x > 0.5:
            t -= 1.0
        return t
    else:
        t = np.ceil(-x)
        if t + x > 0.5:
            t -= 1.0
        return -t
