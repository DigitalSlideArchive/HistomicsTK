import collections
import numpy
import openslide

def TilingSchedule(File, Magnification, Tile):
    '''
    Generates parameters needed to tile a whole-slide-image using OpenSlide 
    for a given resolution 'Magnification' and tilesize 'Tile'.
    *Inputs:
    File (string) - path and filename of slide.
    Magnification (scalar) - desired magnification for analysis.
    Tile (scalar) tilesize at desired magnification.
    *Outputs (tuple):
    Level (scalar) - pyramid level for use with OpenSlide's 'read_region'.
    Scale (scalar) - ratio of 'Magnification' and native scanning magnification.
    Tout (scalar) - tilesize at magnification used for reading, for use with OpenSlide's 'read_region'.
    Factor (scalar) - scaling factor needed for resizing output tiles from 'read_region'.
                      used when desired magnification is not available and downsampling of
                      a higher magnification is necessary.
    X (matrix) - horizontal coordinates for each tile used in calls to 'read_region'.
    X (matrix) - vertical coordinates for each tile used in calls to 'read_region'.
    dX (matrix)  horizontal coordinates at desired magnification. Used
                 for display of boundaries and navigation.
    dY (matrix)  vertical coordinates at desired magnification. Used
                 for display of boundaries and navigation.
    *Related functions:
    '''
    
    #tolerance parameter - acceptable mismatch percentage for desired magnification
    tol = 0.002
    
    #check if slide can be opened
    try:
        Slide = openslide.OpenSlide(File)
    except openslide.OpenSlideError:
        print("Cannot find file '" + File + "'")
        return
    except openslide.OpenSlideUnsupportedFormatError:
        print("Slide format not supported. Consult OpenSlide documentation")
        return
        
    #get slide dimensions, zoom levels, and objective information
    Dims = Slide.level_dimensions
    Factors = Slide.level_downsamples
    Objective = float(Slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
    
    #calculate magnifications
    Available = tuple(Objective / x for x in Factors)
    
    #find highest magnification greater than or equal to 'Desired'
    Mismatch = tuple(x-Magnification for x in Available)
    AbsMismatch = tuple(abs(x) for x in Mismatch)
    if(min(AbsMismatch) <= tol):
        Level = min(AbsMismatch)
        Factor = 1
    else: #pick next highest level, downsample
        Level = max([i for (i, val) in enumerate(Mismatch) if val > 0])
        Factor = Magnification / Available[Level]
        
    #adjust tilesize based on resizing factor
    Tout = round(Tile / Factor)
    
    #generate X, Y coordinates for tiling
    Stride = Tout * Available[0] / Available[Level]
    X = numpy.arange(0, Dims[0][0], Stride)
    Y = numpy.arange(0, Dims[0][1], Stride)
    X, Y = numpy.meshgrid(X, Y)
    dX = X / (Available[0] / Magnification)
    dY = Y / (Available[0] / Magnification)
    
    #calculate scale difference between base and desired magnifications
    Scale = Magnification / Objective
    
    #collect outputs in container
    TilingSchedule = collections.namedtuple('TilingSchedule', ['Level', 'Scale', 'Tout', 'Factor', 'X', 'Y', 'dX', 'dY'])
    Schedule = TilingSchedule(Level, Scale, Tout, Factor, X, Y, dX, dY)
    
    return(Schedule)
