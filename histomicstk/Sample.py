import numpy as np
import openslide
import scipy
from TilingSchedule import TilingSchedule
from ConvertSchedule import ConvertSchedule
from SimpleMask import SimpleMask


def Sample(File, Magnification, Percent, Tile, MappingMag=1.25):
    """Generates a sampling of pixels from a whole-slide image.

    Useful for generating statistics or Reinhard color-normalization or
    adaptive deconvolution. Uses mixture modeling approach to focus
    sampling in tissue regions.

    Parameters
    ----------
    File : str
        path and filename of slide.
    Magnification : double
        Desired magnification for sampling (defaults to native scan
        magnification).
    Percent : double
        Percentage of pixels to sample. Must be in the range [0, 1].
    Tile : int
        Tile size used in sampling high-resolution image.
    MappingMag: double, optional
        low resolution magnification. Default value = 1.25.

    Returns
    -------
    Pixels : array_like
        A 3xN matrix of RGB pixel values sampled from the slide at `File`.

    See Also
    --------
    ReinhardNorm, SparseColorDeconvolution, AdaptiveColorNorm
    """

    # open image
    Slide = openslide.OpenSlide(File)

    # generate tiling schedule for desired sampling magnification
    Schedule = TilingSchedule(File, Magnification, Tile)

    # convert tiling schedule to low-resolution for tissue mapping
    lrSchedule = ConvertSchedule(Schedule, MappingMag)

    # get width, height of image at low-res reading magnification
    lrHeight = Slide.level_dimensions[lrSchedule.Level][1]
    lrWidth = Slide.level_dimensions[lrSchedule.Level][0]

    # NEED TO CHECK lrSchedule.Factor to make sure we don't read a
    # huge buffer in. This would only happen if there are no
    # low-resolution images available

    # read in whole slide at low magnification
    LR = Slide.read_region((0, 0), lrSchedule.Level, (lrWidth, lrHeight))

    # convert to numpy array and strip alpha channel
    LR = np.asarray(LR)
    LR = LR[:, :, :3]

    # resize if desired magnification is not provided by the file
    if lrSchedule.Factor != 1.0:
        LR = scipy.misc.imresize(LR, lrSchedule.Factor)

    # mask
    Mask = SimpleMask(LR)

    # pad mask to match overall size of evenly tiled image
    MaskHeight = lrSchedule.Tout * lrSchedule.Y.shape[0]
    MaskWidth = lrSchedule.Tout * lrSchedule.X.shape[1]
    LRMask = np.zeros((MaskHeight, MaskWidth), dtype=np.uint8)
    LRMask[0:lrHeight, 0:lrWidth] = Mask

    # sample from tile at full resolution that contain more than 1/2 foreground
    Pixels = list()
    for i in range(Schedule.X.shape[0]):
        for j in range(Schedule.X.shape[1]):
            lrTileMask = LRMask[
                i * lrSchedule.Tout:(i + 1) * lrSchedule.Tout,
                j * lrSchedule.Tout:(j + 1) * lrSchedule.Tout].astype(np.uint8)
            TissueCount = sum(lrTileMask.flatten().astype(
                np.float)) / (lrSchedule.Tout**2)
            if TissueCount > 0.5:
                # upsample mask from low-resolution version, and read in color
                # region from desired magnfication
                TileMask = scipy.misc.imresize(lrTileMask,
                                               Schedule.Magnification /
                                               lrSchedule.Magnification,
                                               interp='nearest')
                Tile = Slide.read_region((int(Schedule.X[i, j]),
                                          int(Schedule.Y[i, j])),
                                         Schedule.Level,
                                         (Schedule.Tout, Schedule.Tout))
                Tile = np.asarray(Tile)
                Tile = Tile[:, :, :3]

                # resize if desired magnification is not provided by the file
                if Schedule.Factor != 1.0:
                    Tile = scipy.misc.imresize(Tile, Schedule.Factor)

                # generate linear indices of pixels in mask
                Indices = np.nonzero(TileMask.flatten())[0]
                Sampling = np.random.uniform(0, Indices.size,
                                             (np.ceil(Percent *
                                                      Indices.size),))
                Indices = Indices[Sampling.astype(np.int)]

                # convert rgb tile to 3xN array and sample with linear indices
                Vectorized = np.reshape(Tile,
                                        (Tile.shape[0] * Tile.shape[1], 3))
                Pixels.append(Vectorized[Indices, :].transpose())

    # concatenate pixel values in list
    Pixels = np.concatenate(Pixels, 1)

    return Pixels
