import numpy

def ComplementStainMatrix( W ):
    '''
    Used to fill out empty columns of a stain matrix for use with 
    ColorDeconvolution. Replaces right-most column with normalized cross-product
    of first two columns.
    *Inputs:
        W (matrix) - a 3x3 stain calibration matrix with stain color vectors in columns.
    *Outputs:
        Out(matrix) - a 3x3 complemented stain calibration matrix with a third 
                      orthogonal column.
    *Related functions:
        ColorDeconvolution
    '''
    
    #copy input to output for initialization
    Complemented = W

    #calculatoe directed cross-product of first two columns
    if ((W[0,0]**2 + W[0,1]**2) > 1):
        Complemented[0,2] = 0
    else:
        Complemented[0,2] = (1 - (W[0,0]**2 + W[0,1]**2))**0.5

    if ((W[1,0]**2 + W[1,1]**2) > 1):
        Complemented[1,2] = 0
    else:
        Complemented[1,2] = (1 - (W[1,0]**2 + W[1,1]**2))**0.5
    
    if ((W[2,0]**2 + W[2,1]**2) > 1):
        Complemented[2,2] = 0
    else:
        Complemented[2,2] = (1 - (W[2,0]**2 + W[2,1]**2))**0.5
    
    #normalize new vector to unit-norm
    Complemented[:,2] = Complemented[:,2] / numpy.linalg.norm(Complemented[:,2])
    
    return(Complemented)