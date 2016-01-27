import numpy
import RudermanLABFwd as rlf
import RudermanLABInv as rli

def ReinhardNorm(I, TargetMu, TargetSigma, SourceMu=None, SourceSigma=None):
    '''
    Performs Reinhard color normalization to transform the color characteristics of an image to 
    a desired standard. The standard is defined by the mean and standard deviations of
    the target image in LAB color space defined by Ruderman. The input image is converted to 
    Ruderman's LAB space, the LAB channels are each centered and scaled to zero-mean unit 
    variance, and then rescaled and shifted to match the target image statistics. If the LAB
    statistics for the input image are provided ('SourceMu' and 'SourceSigma') then these
    will be used in the normalization, otherwise they will be derived from the input imagesource
    LAB
    *Inputs:
    	I (rgbimage) - an RGB image of type unsigned char.
    	TargetMu (array) - a 3-element list containing the means of the target image channels in LAB 
    				color space.
    	TargetSigma (array) - a 3-element list containing the standard deviations of the target image
    					channels in LAB color space.
      SourceMu (array, optional) - a 3-element list containing the means of the source image channels in LAB 
    				color space. Used with ReinhardSample for uniform normalization of tiles
                        tiles from a slide.
    	SourceSigma (array, optional) - a 3-element list containing the standard deviations of the source image
    				channels in LAB color space. Used with ReinhardSample for uniform normalization of tiles
                        tiles from a slide.
    *Outputs:
    	Normalized (rgbimage) - a normalized RGB image with corrected color characteristics.
    *Related functions:
    	RudermanLABFwd, RudermanLABInv
    *References:
    Erik Reinhard, Michael Ashikhmin, Bruce Gooch, and Peter Shirley. 2001. Color Transfer between Images. IEEE Comput. Graph. Appl. 21, 5 (September 2001), 34-41. 
    Daniel Ruderman, Thomas Cronin, Chuan-Chin Chiao, Statistics of Cone Responses to Natural Images: Implications for Visual Coding, J. Optical Soc. of America, vol. 15, no. 8, 1998, pp. 2036-2045.
    '''
    
    #get input image dimensions
    m = I.shape[0]
    n = I.shape[1]

    #convert input image to LAB color space
    LAB = rlf.RudermanLABFwd(I)

    #calculate SourceMu if not provided
    if SourceMu is None:
        SourceMu = LAB.sum(axis=0).sum(axis=0) / (m*n)
        print(SourceMu)

    #center to zero-mean
    LAB[:,:,0] = LAB[:,:,0] - SourceMu[0]
    LAB[:,:,1] = LAB[:,:,1] - SourceMu[1]
    LAB[:,:,2] = LAB[:,:,2] - SourceMu[2]

    #calculate SourceSigma if not provided
    if SourceSigma is None:
        SourceSigma = ((LAB*LAB).sum(axis=0).sum(axis=0) / (m*n-1)) ** 0.5
        print(SourceSigma)

    #scale to unit variance
    LAB[:,:,0] = LAB[:,:,0] / SourceSigma[0]
    LAB[:,:,1] = LAB[:,:,1] / SourceSigma[1]
    LAB[:,:,2] = LAB[:,:,2] / SourceSigma[2]

    #rescale and recenter to match target statistics
    LAB[:,:,0] = LAB[:,:,0] * TargetSigma[0] + TargetMu[0]
    LAB[:,:,1] = LAB[:,:,1] * TargetSigma[1] + TargetMu[1]
    LAB[:,:,2] = LAB[:,:,2] * TargetSigma[2] + TargetMu[2]

    #convert back to RGB colorspace
    Normalized = rli.RudermanLABInv(LAB)
    Normalized[Normalized > 255] = 255
    Normalized[Normalized < 0] = 0
    Normalized = Normalized.astype(numpy.uint8)
    
    return(Normalized)