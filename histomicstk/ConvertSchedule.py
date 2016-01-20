import collections
import numpy
import openslide

def ConvertSchedule(Schedule, Magnification):
    '''
    Converts a tiling schedule to a new magnification to generate an equivalent
    tiling.
    *Inputs:
    Schedule (named tuple) - schedule obtained from TilingSchedule.
    Magnification (scalar) - new desired magnification to convert schedule to.
    *Outputs (tuple):
    Level (scalar) - pyramid level for use with OpenSlide's 'read_region'.
    Scale (scalar) - ratio of 'Magnification' and native scanning magnification.
    Tout (scalar) - tilesize at magnification used for reading, for use with OpenSlide's 'read_region'.
    Factor (scalar) - scaling factor needed for resizing output tiles from 'read_region'.
                      used when desired magnification is not available and downsampling of
                      a higher magnification is necessary.
    Magnification (scalar) - magnification of tiling schedule.                      
    X (matrix) - horizontal coordinates for each tile used in calls to 'read_region'.
    X (matrix) - vertical coordinates for each tile used in calls to 'read_region'.
    dX (matrix)  horizontal coordinates at desired magnification. Used
                 for display of boundaries and navigation.
    dY (matrix)  vertical coordinates at desired magnification. Used
                 for display of boundaries and navigation.
    *Related functions:
    TilingSchedule
    '''
    
    #tolerance parameter - acceptable mismatch percentage for desired magnification
    tol = 0.002
    
    #check if slide can be opened
    try:
        Slide = openslide.OpenSlide(Schedule.File)
    except openslide.OpenSlideError:
        print("Cannot find file '" + Schedule.File + "'")
        return
    except openslide.OpenSlideUnsupportedFormatError:
        print("Slide format not supported. Consult OpenSlide documentation")
        return
    
    #get slide dimensions, zoom levels, and objective information
    Dims = Slide.level_dimensions
    Factors = Slide.level_downsamples
    Objective = float(Slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])    
    
    #determine if desired magnification is avilable in file
    Available = tuple(Objective / x for x in Factors)
    Mismatch = tuple(x-Magnification for x in Available)
    AbsMismatch = tuple(abs(x) for x in Mismatch)
    if(min(AbsMismatch) <= tol):
        Level = int(AbsMismatch.index(min(AbsMismatch)))
        Factor = 1
    else: #pick next highest level, downsample
        Level = int(max([i for (i, val) in enumerate(Mismatch) if val > 0]))
        Factor = Magnification / Available[Level]

    #translate parameters of input tiling schedule into new schedule
    Tout = int(round(((Schedule.Tout * Schedule.Factor) / Schedule.Magnification) * Magnification / Factor))
    
    Stride = Tout * Available[0] / Available[Level]
    X = Schedule.X
    Y = Schedule.Y
    dX = Schedule.dX
    dY = Schedule.dY
    
    #calculate scale difference between base and desired magnifications
    Scale = Magnification / Objective
    
    #collect outputs in container
    TilingSchedule = collections.namedtuple('TilingSchedule', 
                                            ['Level', 'Scale', 'Tout', 'Factor',
                                             'Magnification', 'File', 'X', 'Y',
                                             'dX', 'dY'])
    Converted = TilingSchedule(Level, Scale, Tout, Factor, Magnification, Schedule.File, X, Y, dX, dY)
    
    return(Converted)