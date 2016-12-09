import numpy as np
import openslide
import scipy

from .tiling_schedule import tiling_schedule
from .convert_schedule import convert_schedule
from .simple_mask import simple_mask


def sample(slide_path, magnification, percent, tile_size,
           mapping_mag=1.25, min_coverage=0.1):
    """Generates a sampling of pixels from a whole-slide image.

    Useful for generating statistics or Reinhard color-normalization or
    adaptive deconvolution. Uses mixture modeling approach to focus
    sampling in tissue regions.

    Parameters
    ----------
    slide_path : str
        path and filename of slide.
    magnification : double
        Desired magnification for sampling (defaults to native scan
        magnification).
    percent : double
        Percentage of pixels to sample. Must be in the range [0, 1].
    tile_size : int
        Tile size used in sampling high-resolution image.
    mapping_mag: double, optional
        low resolution magnification. Default value = 1.25.
    min_coverage: double, optional
        minimum percent of tile covered by tissue to be included in sampling.
        Ranges between [0,1). Default value = 0.1.

    Returns
    -------
    pixels : array_like
        A 3xN matrix of RGB pixel values sampled from the slide at `File`.

    See Also
    --------
    histomicstk.preprocessing.color_normalization.reinhard,
    histomicstk.preprocessing.color_deconvolution.SparseColorDeconvolution
    """

    # open image
    Slide = openslide.OpenSlide(slide_path)

    # generate tiling schedule for desired sampling magnification
    Schedule = tiling_schedule(slide_path, magnification, tile_size)

    # convert tiling schedule to low-resolution for tissue mapping
    lrSchedule = convert_schedule(Schedule, mapping_mag)

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
        lrHeight = LR.shape[0]
        lrWidth = LR.shape[1]

    # mask
    Mask = simple_mask(LR)

    # pad mask to match overall size of evenly tiled image
    MaskHeight = lrSchedule.Tout * lrSchedule.Y.shape[0]
    MaskWidth = lrSchedule.Tout * lrSchedule.X.shape[1]
    LRMask = np.zeros((MaskHeight, MaskWidth), dtype=np.uint8)
    LRMask[0:lrHeight, 0:lrWidth] = Mask

    # sample from tile at full resolution that contain more than 1/2 foreground
    pixels = list()
    for i in range(Schedule.X.shape[0]):
        for j in range(Schedule.X.shape[1]):
            lrTileMask = LRMask[
                i * lrSchedule.Tout:(i + 1) * lrSchedule.Tout,
                j * lrSchedule.Tout:(j + 1) * lrSchedule.Tout].astype(np.uint8)
            TissueCount = sum(lrTileMask.flatten().astype(
                np.float)) / (lrSchedule.Tout**2)
            if TissueCount > min_coverage:
                # region from desired magnfication
                tile_size = Slide.read_region((int(Schedule.X[i, j]),
                                               int(Schedule.Y[i, j])),
                                              Schedule.Level,
                                              (Schedule.Tout, Schedule.Tout))
                tile_size = np.asarray(tile_size)
                tile_size = tile_size[:, :, :3]

                # resize if desired magnification is not provided by the file
                if Schedule.Factor != 1.0:
                    tile_size = scipy.misc.imresize(tile_size,
                                                    Schedule.Factor,
                                                    interp='nearest')

                # upsample tile mask from low-resolution to high-resolution
                TileMask = scipy.misc.imresize(lrTileMask,
                                               tile_size.shape,
                                               interp='nearest')

                # generate linear indices of pixels in mask
                Indices = np.nonzero(TileMask.flatten())[0]
                Sampling = np.random.uniform(0, Indices.size,
                                             (np.ceil(percent *
                                                      Indices.size),))
                Indices = Indices[Sampling.astype(np.int)]

                # convert rgb tile to 3xN array and sample with linear indices
                Vectorized = np.reshape(tile_size,
                                        (tile_size.shape[0] * tile_size.shape[1], 3))
                pixels.append(Vectorized[Indices, :].transpose())

    # concatenate pixel values in list
    try:
        pixels = np.concatenate(pixels, 1)
    except ValueError:
        print "Sampling could not identify any foreground regions."

    return pixels
