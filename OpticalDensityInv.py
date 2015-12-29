import numpy

def OpticalDensityInv( I ):
    '''
    Transforms input RGB image "I" into optical density space for color deconvolution.
    *Inputs:
        I (rgbimage) - a floating-point image of optical density values obtained
                        from OpticalDensityFwd.
    *Outputs:
        Out (rgbimage) - a floating-point multi-channel intensity image with 
                        values in range 0-255.
    *Related functions:
        OpticalDensityFwd, ColorDeconvolution, ColorConvolution                     
    '''
    
    return numpy.exp(-(I - 255)*numpy.log(255)/255);
