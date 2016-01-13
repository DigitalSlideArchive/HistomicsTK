import numpy

def RudermanLABInv(I):
    '''
    Transforms an LAB image into RGB space defined by Ruderman in the 1998 paper.
    *Inputs:
        LAB (rgbimage) - an LAB image of type unsigned char.
    *Outputs:
        LAB (rgbimage) - the RGB representation of the input image 'I'.
    *Related functions:
        RudermanLABFwd, ReinhardNorm
    *References:
        "Statistics of Cone Responses to Natural Images: Implications for Visual Coding" J. Optical Society of America, vol. 15, no. 8, 1998, pages 2036-45.
    '''

    #get input image dimensions
    m = I.shape[0]
    n = I.shape[1]
    
    #define conversion matrices
    LAB2LMS = numpy.array([[1,1,1],[1,1,-1],[1,-2,0]]).dot(numpy.array([[1/(3**(0.5)),0,0],[0,1/(6**(0.5)),0],[0,0,1/(2**(0.5))]]))
    LMS2RGB = numpy.array([[4.4679,-3.5873,0.1193],[-1.2186,2.3809,-0.1624],[0.0497,-0.2439,1.2045]])
    
    #calculate LMS values from LAB
    I = numpy.reshape(I, (m*n,3))
    LMS = numpy.dot(LAB2LMS, numpy.transpose(I))
    expLMS = numpy.exp(LMS)
    
    #calculate RGB values from LMS
    RGB = LMS2RGB.dot(expLMS)
    
    #reshape to 3-channel image
    RGB = numpy.reshape(RGB.transpose(), (m,n,3))
    
    return(RGB)