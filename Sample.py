import numpy
import openslide
import scipy

def Sample(File, Magnification, Percent, Tile):
    '''
    Generates a sampling of pixels from a whole-slide image. Useful for generating statistics
    for Reinhard color-normalization or adaptive deconvolution. Uses mixture modeling approach
    to focus sampling in tissue regions.
    *Inputs:
        File (string) - path and filename of slide.
        Magnification (scalar) - desired magnification for sampling (defaults to native scan magnification).
        Percent (scalar) - percentage of pixels to sample (range (0, 1]).
        Tile (scalar) - tile size used in sampling high-resolution image.
    *Outputs:
        Pixels (matrix) - a 3xN matrix of RGB pixel values sampled from the slide at 'File'.
    *Related functions:
        ReinhardNorm, SparseColorDeconvolution, AdaptiveColorNorm
    *References:
    '''
    
    #parameters
    MappingMag = 1.25  #low resolution magnification

    #open image
    Slide = openslide.OpenSlide(File)

    #generate tiling schedule for desired sampling magnification
    Schedule = TilingSchedule(File, Magnification, Tile)

    #convert tiling schedule to low-resolution for tissue mapping
    lrSchedule = ConvertSchedule(Schedule, MappingMag)
    lrHeight = lrSchedule.Tout * lrSchedule.Y.shape[0]
    lrWidth = lrSchedule.Tout * lrSchedule.X.shape[1]

    #NEED TO CHECK lrSchedule.Factor to make sure we don't read a huge buffer in.
    #This would only happen if there are no low-resolution images available

    #read in whole slide at low magnification
    LR = Slide.read_region((0, 0), lrSchedule.Level, (lrWidth, lrHeight))

    #convert to numpy array and strip alpha channel
    LR = numpy.asarray(LR)
    LR = LR[:,:,:3]

    #resize if desired magnification is not provided by the file
    if(lrSchedule.Factor != 1.0):
        LR = scipy.misc.imresize(LR, lrSchedule.Factor)

    #mask
    LRMask = SimpleMask(LR)

    #sample from tile at full resolution that contain more than 1/2 foreground
    Pixels = list()
    for i in range(Schedule.X.shape[0]):
        for j in range(Schedule.X.shape[1]):
            lrTileMask = LRMask[i*lrSchedule.Tout:(i+1)*lrSchedule.Tout,
                                j*lrSchedule.Tout:(j+1)*lrSchedule.Tout].astype(numpy.uint8)
            TissueCount = sum(lrTileMask.flatten().astype(numpy.float)) / (lrSchedule.Tout**2)
            if(TissueCount > 0.5):
                #upsample mask from low-resolution version, and read in color region from desired magnfication
                TileMask = scipy.misc.imresize(lrTileMask, Schedule.Magnification / lrSchedule.Magnification,
                                               interp='nearest')
                Tile = Slide.read_region((int(Schedule.X[i,j]), int(Schedule.Y[i,j])), Schedule.Level, (Schedule.Tout, Schedule.Tout))
                Tile = numpy.asarray(Tile)
                Tile = Tile[:,:,:3]

                #resize if desired magnification is not provided by the file
                if(Schedule.Factor != 1.0):
                    Tile = scipy.misc.imresize(Tile, Schedule.Factor)

                #generate linear indices of pixels in mask
                Indices = numpy.nonzero(TileMask.flatten())[0]
                Sampling = numpy.random.uniform(0, Indices.size, (numpy.ceil(Percent*Indices.size),))
                Indices = Indices[Sampling.astype(numpy.int)]

                #convert rgb tile to 3 x N array and sample with linear indices
                Vectorized = numpy.reshape(Tile, (Tile.shape[0]*Tile.shape[1],3))
                Pixels.append(Vectorized[Indices, :].transpose())

    #concatenate pixel values in list
    Pixels = numpy.concatenate(Pixels, 1)
        
    return(Pixels)