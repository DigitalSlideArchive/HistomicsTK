import numpy as np
import scipy.ndimage.measurements as spm


def MaxClustering(Response, Mask, r=10):
    """Local max clustering pixel aggregation for nuclear segmentation.

    Takes as input a constrained log or other filtered nuclear image, a binary 
    nuclear mask, and a clustering radius. For each pixel in the nuclear mask, 
    the local max is identified. A hierarchy of local maxima is defined, and 
    the root nodes used to define the label image.

    Parameters
    ----------
    Response : array_like
        A filtered-smoothed image where the maxima correspond to nuclear
        center. Typically obtained by constrained-LoG filtering on a 
        hematoxylin intensity image obtained from ColorDeconvolution. 
    Mask : array_like
        A binary image where nuclei pixels have value 1/True, and non-nuclear
        pixels have value 0/False.
    f : float
        A scalar defining the clustering radius. Default value = 10.

    Returns
    -------
    Label : array_like
        Label image where positive values correspond to foreground pixels that
        share mutual sinks.
    Seeds : array_like
        An N x 2 array defining the (x,y) coordinates of nuclei seeds.
    Max : array_like
        An intensity image where each pixel corresponds to the local max of
        'Response' in the 'r' radius neighborhood.

    See Also
    --------
    cLoG

    References
    ----------
    .. [1] XW. Wu et al “The local maximum clustering method and its 
    application in microarray gene expression data analysis,” EURASIP J. Appl. 
    Signal Process,vol.2004,no.1,pp.53–63, 2004.
    .. [2] Y. Al-Kofahi et al "Improved Automatic Detection and Segmentation
    of Cell Nuclei in Histopathology Images" in IEEE Transactions on Biomedical
    Engineering,vol.57,no.4,pp.847-52, 2010.
    """

    # define kernel for max filter
    Kernel = np.zeros((2*r+1, 2*r+1), dtype=bool)
    X, Y = np.meshgrid(np.linspace(0, 2*r, 2*r+1), np.linspace(0, 2*r, 2*r+1))
    X -= r
    Y -= r
    Kernel[(X**2 + Y**2)**0.5 <= r] = True

    # define linear coordinates of postive kernel entries
    X = X[Kernel].astype(np.int)
    Y = Y[Kernel].astype(np.int)

    # pad input array to simplify filtering
    I = Response.min() * np.ones((Response.shape[0]+2*r,
                                  Response.shape[1]+2*r))
    Response[~Mask] = Response.min()
    I[r:r+Response.shape[0], r:r+Response.shape[1]] = Response

    # initialize coordinate arrays and max value arrays
    Max = np.zeros(I.shape)
    Row = np.zeros(I.shape, dtype=np.int)
    Col = np.zeros(I.shape, dtype=np.int)

    # define pixels for local neighborhoods
    py, px = np.nonzero(Mask)
    py = py + r
    px = px + r

    # perform max filtering
    for i in np.arange(0, px.size, 1):

        # calculate local max value and position around px[i], py[i]
        Index = np.argmax(I[py[i]+Y, px[i]+X])
        Max[py[i], px[i]] = I[py[i]+Y[Index], px[i]+X[Index]]
        Row[py[i], px[i]] = py[i] + Y[Index] - r
        Col[py[i], px[i]] = px[i] + X[Index] - r

    # trim outputs
    Max = Max[r:Response.shape[0]+r, r:Response.shape[1]+r]
    Row = Row[r:Response.shape[0]+r, r:Response.shape[1]+r]
    Col = Col[r:Response.shape[0]+r, r:Response.shape[1]+r]

    # subtract out padding offset for px, py
    py = py - r
    px = px - r

    # initialize tracking and segmentation masks
    Tracked = np.zeros(Max.shape, dtype=bool)
    Label = spm.label((Response == Max) & Mask)[0]
    Tracked[Label > 0] = True
    Seeds = np.reshape(np.nonzero(Label),
                       [2, Label.max()], order='C').transpose()

    # track each pixel and update
    for i in np.arange(0, px.size, 1):

        # initialize tracking trajectory
        Id = 0
        Alloc = 1
        Trajectory = np.zeros((1000, 2), dtype=np.int)
        Trajectory[0, 0] = px[i]
        Trajectory[0, 1] = py[i]

        while(~Tracked[Trajectory[Id, 1], Trajectory[Id, 0]]):

            # increment trajectory counter
            Id += 1

            # if overflow, copy and reallocate
            if(Id == 1000*Alloc):
                temp = Trajectory
                Trajectory = np.zeros((1000*(Alloc+1), 2), dtype=np.int)
                Trajectory[0:1000*(Alloc), ] = temp
                Alloc += 1

            # add local max to trajectory
            Trajectory[Id, 0] = Col[Trajectory[Id-1, 1], Trajectory[Id-1, 0]]
            Trajectory[Id, 1] = Row[Trajectory[Id-1, 1], Trajectory[Id-1, 0]]

        # label sequence and add to tracked list
        Tracked[Trajectory[0:Id, 1], Trajectory[0:Id, 0]] = True
        Label[Trajectory[0:Id, 1], Trajectory[0:Id, 0]] = \
            Label[Trajectory[Id, 1], Trajectory[Id, 0]]

    # return
    return Label, Seeds, Max
