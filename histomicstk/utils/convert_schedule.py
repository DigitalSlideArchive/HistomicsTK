import collections
import openslide


def convert_schedule(schedule, magnification, tol=0.002):
    """Converts a tiling schedule to a new magnification

    Parameters
    ----------
    schedule : collections.namedtuple
        schedule obtained from TilingSchedule.
    magnification : scalar
        Desired magnification to convert schedule to.
    tol : double, optional
        Acceptable mismatch percentage for desired magnification.
        Default value is 0.002.

    Returns
    -------
    Level : int
        pyramid level for use with OpenSlide's 'read_region'.
    Scale : double
        ratio of 'Magnification' and native scanning magnification.
    Tout : scalar
        tilesize at magnification used for reading, for use with OpenSlide's
        'read_region'.
    Factor : double
        scaling factor needed for resizing output tiles from 'read_region'.
        used when desired magnification is not available and downsampling of
        a higher magnification is necessary.
    Magnification : scalar
        magnification of tiling schedule.
    X : array_like
        horizontal coordinates for each tile used in calls to 'read_region'.
    Y : array_like
        vertical coordinates for each tile used in calls to 'read_region'.
    dX : array_like
        horizontal coordinates at desired magnification. Used for display of
        boundaries and navigation.
    dY : array_like
        vertical coordinates at desired magnification. Used for display of
        boundaries and navigation.

    Notes
    -----
    The converted schedule is returned as a named tuple.

    See Also
    --------
    histomicstk.utils.tiling_schedule
    """

    # check if slide can be opened
    try:
        Slide = openslide.OpenSlide(schedule.File)
    except openslide.OpenSlideError:
        print("Cannot find file '" + schedule.File + "'")
        return
    except openslide.OpenSlideUnsupportedFormatError:
        print("Slide format not supported. Consult OpenSlide documentation")
        return

    # get slide dimensions, zoom levels, and objective information
    Dims = Slide.level_dimensions  # noqa
    Factors = Slide.level_downsamples
    Objective = float(Slide.properties[
        openslide.PROPERTY_NAME_OBJECTIVE_POWER])

    # determine if desired magnification is avilable in file
    Available = tuple(Objective / x for x in Factors)
    Mismatch = tuple(x - magnification for x in Available)
    AbsMismatch = tuple(abs(x) for x in Mismatch)
    if min(AbsMismatch) <= tol:
        Level = int(AbsMismatch.index(min(AbsMismatch)))
        Factor = 1
    else:
        # pick next highest level, downsample
        Level = int(max([i for (i, val) in enumerate(Mismatch) if val > 0]))
        Factor = magnification / Available[Level]

    # translate parameters of input tiling schedule into new schedule
    Tout = int(round(((schedule.Tout * schedule.Factor) /
                      schedule.Magnification) * magnification / Factor))

    Stride = Tout * Available[0] / Available[Level]  # noqa
    X = schedule.X
    Y = schedule.Y
    dX = schedule.dX
    dY = schedule.dY

    # calculate scale difference between base and desired magnifications
    Scale = magnification / Objective

    # collect outputs in container
    TilingSchedule = collections.namedtuple('TilingSchedule',
                                            ['Level', 'Scale',
                                             'Tout', 'Factor',
                                             'Magnification', 'File', 'X', 'Y',
                                             'dX', 'dY'])

    conv_schedule = TilingSchedule(Level,
                                   Scale,
                                   Tout,
                                   Factor,
                                   magnification,
                                   schedule.File,
                                   X, Y, dX, dY)

    return conv_schedule
