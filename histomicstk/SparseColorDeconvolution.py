import collections
import nimfa
import numpy
import OpticalDensityFwd as odf

def SparseColorDeconvolution (I, Winit=None, Beta=0.5):
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
        I (rgbimage) - an RGB image of type unsigned char, or a 3xN matrix of
                        RGB pixel values of type unsigned char, where each column
                        is a pixel and each row a color channel.
        Winit (matrix, optional) - a 3xK matrix containing the color vectors in 
                    columns. Should not be complemented with ComplementStainMatrix 
                    for sparse decomposition to work correctly.
        Beta (scalar) - regularization factor for sparsity of "H" - recommended 0.5
    *Outputs (tuple):
        Stains (rgbimage) - an rgb image or KxN matrix with deconvolved stain intensities in 
                            each channel, values ranging from [0, 255], suitable for display.
        W (matrix) - the final 3 x K stain matrix produced by the NMF decomposition.
    '''

    #determine if image is in m x n x 3 RGB image or 3 x N matrix format       
    if(len(I.shape) == 3): #RBG image provided
        m = I.shape[0]
        n = I.shape[1]
        I = numpy.reshape(I, (m*n,3)).transpose()
    elif(len(I.shape) == 2): #pixel matrix provided
        m = -1
        n = -1
        if(I.shape[2] == 4): #remove alpha channel if needed
            I = I[:,:,(0,1,2)]
    
    #transform input RGB to optical density values
    I = I.astype(dtype=numpy.float32)
    I[I == 0] = 1e-16
    ODfwd = odf.OpticalDensityFwd(I)

    if (Winit is None):
        
        #set number of output stains
        K = 3
        
        #perform NMF without initialization
        Factorization = nimfa.Snmf(V=ODfwd, seed=None, rank = K, version='r', beta=Beta)
        Factorization()

    else:
        
        #get number of output stains
        K = Winit.shape[1]
        
        #normalize stains to unit-norm
        for i in range(K):
            Norm = numpy.linalg.norm(Winit[:,i])
            if(Norm >= 1e-16):
                Winit[:,i] /= Norm
            else:
                print 'error' #throw error
    
        #estimate initial H given p
        Hinit = numpy.dot(numpy.linalg.pinv(Winit), ODfwd)
        Hinit[Hinit < 0] = 0
    
        #perform NMF
        Factorization = nimfa.Snmf(V=ODfwd, seed=None, W = Winit, H = Hinit, rank = K, version='r', beta=Beta)
        Factorization()


    #extract solutions and make columns of "W" unit-norm
    W = numpy.asarray(Factorization.basis())
    H = numpy.asarray(Factorization.coef())
    for i in range(K):
        Norm = numpy.linalg.norm(W[:,i])
        W[:,i] /= Norm
        H[i,:] *= Norm
        
    #reshape H matrix to image
    if(m == -1):
        StainsFloat = numpy.transpose(H)
    else:
        StainsFloat = numpy.reshape(numpy.transpose(H), (m,n,K))
    
    #transform type  
    Stains = numpy.copy(StainsFloat)
    Stains[Stains > 255] = 255
    Stains = Stains.astype(numpy.uint8)
    
    #build named tuple for outputs
    Unmixed = collections.namedtuple('Unmixed', ['Stains', 'W'])
    Output = Unmixed(Stains, W)
    
    #return solution
    return(Output)