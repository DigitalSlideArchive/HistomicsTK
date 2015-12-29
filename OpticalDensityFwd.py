import numpy

def OpticalDensityFwd(I):
    '''
    Transforms input RGB image "I" into optical density space for color deconvolution.
    *Inputs:
        I (rgbimage) - a floating-point RGB image with channel ranges 0-255.
    *Outputs:
        Out (rgbimage) - a floating-point image of corresponding optical 
                         density values.
    *Related functions:
        OpticalDensityInv, ColorDeconvolution, ColorConvolution
    '''
    
    return (-(255*numpy.log(I/255))/numpy.log(255));
