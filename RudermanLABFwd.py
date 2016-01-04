import numpy

def RudermanLABFwd(I):
    '''
    Transforms an RGB color image into the LAB space defined by Ruderman in the 1998 paper. 
    *Inputs:
        I (rgbimage) - an RGB image of type unsigned char.
    *Outputs:
        LAB (rgbimage) - the LAB representation of the input image 'I'.
    *Related functions:
        RudermanLABInv, ReinhardNorm
    *References:
        "Statistics of Cone Responses to Natural Images: Implications for Visual Coding" J. Optical Society of America, vol. 15, no. 8, 1998, pages 2036-45.
    '''

    #get input image dimensions
    m = I.shape[0]
    n = I.shape[1]

    #define conversion matrices
    RGB2LMS = numpy.array([[0.3811,0.5783,0.0402],[0.1967,0.7244,0.0782],[0.0241,0.1288,0.8444]])
    LMS2LAB = numpy.array([[1/(3**(0.5)),0,0],[0,1/(6**(0.5)),0],[0,0,1/(2**(0.5))]]).dot(numpy.array([[1,1,1],[1,1,-2],[1,-1,0]]))

    #calculate LMS values from RGB
    I = numpy.reshape(I, (m*n,3))
    LMS = numpy.dot(RGB2LMS, numpy.transpose(I))
    logLMS = numpy.log(LMS)

    #calculate LAB values from LMS
    LAB = LMS2LAB.dot(logLMS)

    #reshape to 3-channel image
    LAB = numpy.reshape(LAB.transpose(), (m,n,3))

    return(LAB)