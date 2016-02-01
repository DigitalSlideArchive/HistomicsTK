import collections
import numpy
import ComplementStainMatrix as csm
import OpticalDensityFwd as odf
import OpticalDensityInv as odi

def ColorDeconvolution(I, W):
    '''
    Implements color deconvolution. The input image "I" consisting of RGB values is
    first transformed into optical density space, and then projected onto the stain
    vectors in the columns of "W".
    example H&E matrix
    W =array([[0.650, 0.072, 0],
              [0.704, 0.990, 0],
              [0.286, 0.105, 0]])
    *Inputs:
        I (rgbimage) - an RGB image of type unsigned char.
        W (matrix) - a 3x3 matrix containing the color vectors in columns. 
                     For two stain images the third column is zero and will be 
                     complemented using cross-product. Minumum two nonzero 
                     columns required.
    *Outputs:
        Stains (rgbimage) - an rgb image with deconvolved stain intensities in 
                            each channel, values ranging from [0, 255], suitable for display.
        StainsFloat (matrix) - an intensity image of deconvolved stains that is 
                            unbounded, suitable for reconstructing color images 
                            of deconvolved stains with ColorConvolution.
        Wc (matrix) - a 3x3 complemented stain matrix. Useful for color image
                      reconstruction with ColorConvolution.
    *Related functions:
        ComplementStainMatrix, OpticalDensityFwd, OpticalDensityInv, ColorConvolution
    '''

    #complement stain matrix if needed
    if(numpy.linalg.norm(W[:,2]) <= 1e-16):
        Wc = csm.ComplementStainMatrix(W)
    else:
        Wc = W.copy()            

    #normalize stains to unit-norm
    for i in range(Wc.shape[1]):
        Norm = numpy.linalg.norm(Wc[:,i])
        if(Norm >= 1e-16):
            Wc[:,i] /= Norm

    #invert stain matrix
    Q = numpy.linalg.inv(Wc)

    #transform 3D input image to 2D RGB matrix format
    m = I.shape[0]
    n = I.shape[1]
    if(I.shape[2] == 4):
        I = I[:,:,(0,1,2)]
    I = numpy.reshape(I, (m*n,3))

    #transform input RGB to optical density values and deconvolve, tfm back to RGB
    I = I.astype(dtype=numpy.float32)
    I[I == 0] = 1e-16
    ODfwd = odf.OpticalDensityFwd(I)
    ODdeconv = numpy.dot(ODfwd, numpy.transpose(Q))
    ODinv = odi.OpticalDensityInv(ODdeconv)

    #reshape output
    StainsFloat = numpy.reshape(ODinv, (m,n,3))
    
    #transform type  
    Stains = numpy.copy(StainsFloat)
    Stains[Stains > 255] = 255
    Stains = Stains.astype(numpy.uint8)
    
    #return 
    Unmixed = collections.namedtuple('Unmixed', ['Stains', 'StainsFloat', 'Wc'])
    Output = Unmixed(Stains, StainsFloat, Wc)
    
    return (Output)
