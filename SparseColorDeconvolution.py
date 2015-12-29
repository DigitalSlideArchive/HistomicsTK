import nimfa
import numpy
import collections

def SparseColorDeconvolution (I, Winit, Beta):
    '''
    Uses sparse non-negative matrix factorization to adaptively deconvolve the 
    color RGB image "I" into intensity images representing distinct stains. 
    Similar approach to ColorDeconvolution but operates adaptively. The input 
    image "I" consisting of RGB values is first transformed into optical density 
    space as a row-matrix, and then is decomposed as V = WH where "W" is a 
    3xk matrix containing stain vectors in columns and "H" is a k x m*n matrix
    of concentrations for each stain vector. The system is solved to encourage 
    sparsity of the columns of "H" - that is most pixels should not contain 
    significant contributions from more than one stain. Can use a hot-start 
    initialization from a color deconvolution matrix.
    *Inputs:
        I (rgbimage) - an RGB image of type unsigned char.
        Winit (matrix, optional) - a 3xK matrix containing the color vectors in 
                    columns. Should not be complemented with ComplementStainMatrix 
                    for sparse decomposition to work correctly.
        Beta (scalar) - regularization factor for sparsity of "H".
    *Outputs (tuple):
        W (matrix) - the final 3 x K stain matrix produced by the NMF decomposition.
        H (matrix) - the corresponding K x N coefficient matrix of deconvolved 
                     stain intensities for each pixel (pixels in columns, stains
                     in rows).
    '''
    
    #get number of output stains
    K = Winit.shape[1]
    
    #normalize stains to unit-norm
    for i in range(K):
        Norm = numpy.linalg.norm(Winit[:,i])
        if(Norm >= 1e-16):
            Winit[:,i] /= Norm
        else:
            print 'error' #throw error
    
    #transform 3D input image to 2D RGB matrix format
    m = I.shape[0]
    n = I.shape[1]
    if(I.shape[2] == 4):
        I = I[:,:,(0,1,2)]
    I = numpy.reshape(I, (m*n,3)).transpose()
    
    #transform input RGB to optical density values
    I = I.astype(dtype=numpy.float32)
    I[I == 0] = 1e-16
    ODfwd = OpticalDensityFwd(I)
    
    #estimate initial H given p
    Hinit = numpy.dot(numpy.linalg.pinv(Winit), ODfwd)
    Hinit[Hinit < 0] = 0
    
    #perform NMF
    Factorization = nimfa.Snmf(V=ODfwd, seed=None, W = Winit, H = Hinit, rank = K, version='r', beta=Beta)
    Factorization()
    
    #extract solutions and make columns of "W" unit-norm
    W = Factorization.basis()
    H = Factorization.coef()
    for i in range(k):
        Norm = numpy.linalg.norm(W[:,i])
        W[:,i] /= Norm
        H[i,:] *= Norm
        
    #build named tuple for outputs
    SparseColorDecomp = collections.namedtuple('SparseColorDecomp', ['W', 'H'])
    Output = SparseColorDecomp(W, H)
    
    #return solution
    return(Output)
