import numpy
import matplotlib.pyplot as plt
from skimage import color
from sklearn.neighbors.kde import KernelDensity
from scipy.stats import norm
from scipy.optimize import fmin_slsqp
from scipy import signal

def SimpleMask(I):
    '''
    Uses a simple two-component Gaussian mixture model to mask tissue areas from 
    background in brightfield H&E images. Kernel-density estimation is used to 
    create a smoothed image histogram, and then this histogram is analyzed to 
    identify modes corresponding to tissue and background. The mode peaks are 
    then analyzed to estimate their width, and a constrained optimization is 
    performed to fit gaussians directly to the histogram (instead of using 
    expectation-maximization directly on the data which is more prone to local
    minima effects). A maximum-likelihood threshold is then derived and used 
    to mask the tissue area in a binarized image.
    *Inputs:
        I (rgb) - an RGB image of type unsigned char.
    *Outputs:
        Mask (binary) - a binarized version of 'I' where foreground (tissue) has value '1'.
    *Related functions:
    Sample
    *References:
    '''
    
    #parameters
    BW = 2  #bandwidth for kernel density estimation - used for tissue segmentation
    MinPeak = 10 #peak widths for finding identifying peaks in KDE histogram - used in initializing curve fitting process
    MaxPeak = 25
    DefaultBGScale = 2.5 #default standard deviation of background gaussian, used if estimation fails
    DefaultTissueScale = 30 #default standard deviation of tissue gaussian, used if estimation fails
    MinProb = 0.05 #minimum probability to qualify as tissue pixel

    #convert image to grayscale, flatten and sample
    I = 255*color.rgb2gray(I)
    I = I.astype(numpy.uint8)
    sI = I.flatten()[:, numpy.newaxis]
    sI = sI[numpy.random.uniform(1, sI.size, (Percent*I.size,)).astype(int)]

    #kernel-density smoothed histogram
    KDE = KernelDensity(kernel='gaussian', bandwidth=BW).fit(sI)
    xHist = numpy.linspace(0,255,256)[:, numpy.newaxis]
    yHist = numpy.exp(KDE.score_samples(xHist))[:, numpy.newaxis]
    yHist = yHist / sum(yHist)

    #flip smoothed y-histogram so that background mode is on the left side
    yHist = numpy.flipud(yHist)

    #identify initial mean parameters for gaussian mixture distribution
    Peaks = signal.find_peaks_cwt(yHist.flatten(), numpy.arange(MinPeak, MaxPeak))
    BGPeak = Peaks[0]
    TissuePeak = Peaks[yHist[Peaks[1:]].argmax()+1] #take highest peak among remaining peaks as background

    #analyze background peak to estimate variance parameter via FWHM
    Left = BGPeak
    while(yHist[Left] > yHist[BGPeak] / 2 and Left >= 0):
        Left = Left-1
        if(Left == -1):
            break
    Right = BGPeak
    while(yHist[Right] > yHist[BGPeak] / 2 and Right < yHist.size):
        Right = Right+1
        if(Right == yHist.size):
            break
    if(Left != -1 and Right != yHist.size):
        LeftSlope = yHist[Left+1]-yHist[Left] / (xHist[Left+1]-xHist[Left])
        Left = (yHist[BGPeak] / 2 - yHist[Left]) / LeftSlope + xHist[Left]
        RightSlope = yHist[Right]-yHist[Right-1] / (xHist[Right]-xHist[Right-1])
        Right = (yHist[BGPeak] / 2 - yHist[Right]) / RightSlope + xHist[Right]
        BGScale = (Right-Left) / 2.355
    if(Left == -1):
        if(Right == yHist.size):
            BGScale = DefaultBGScale
        else:
            RightSlope = yHist[Right]-yHist[Right-1] / (xHist[Right]-xHist[Right-1])
            Right = (yHist[BGPeak] / 2 - yHist[Right]) / RightSlope + xHist[Right]
            BGScale = 2*(Right-xHist[BGPeak]) / 2.355
    if(Right == yHist.size):
        if(Left == -1):
            BGScale = DefaultBGScale
        else:
            LeftSlope = yHist[Left+1]-yHist[Left] / (xHist[Left+1]-xHist[Left])
            Left = (yHist[BGPeak] / 2 - yHist[Left]) / LeftSlope + xHist[Left]
            BGScale = 2*(xHist[BGPeak]-Left) / 2.355

    #analyze tissue peak to estimate variance parameter via FWHM
    Left = TissuePeak
    while(yHist[Left] > yHist[TissuePeak] / 2 and Left >= 0):
        Left = Left-1
        if(Left == -1):
            break
    Right = TissuePeak
    while(yHist[Right] > yHist[TissuePeak] / 2 and Right < yHist.size):
        Right = Right+1
        if(Right == yHist.size):
            break
    if(Left != -1 and Right != yHist.size):
        LeftSlope = yHist[Left+1]-yHist[Left] / (xHist[Left+1]-xHist[Left])
        Left = (yHist[TissuePeak] / 2 - yHist[Left]) / LeftSlope + xHist[Left]
        RightSlope = yHist[Right]-yHist[Right-1] / (xHist[Right]-xHist[Right-1])
        Right = (yHist[TissuePeak] / 2 - yHist[Right]) / RightSlope + xHist[Right]
        TissueScale = (Right-Left) / 2.355
    if(Left == -1):
        if(Right == yHist.size):
            TissueScale = DefaultTissueScale
        else:
            RightSlope = yHist[Right]-yHist[Right-1] / (xHist[Right]-xHist[Right-1])
            Right = (yHist[TissuePeak] / 2 - yHist[Right]) / RightSlope + xHist[Right]
            TissueScale = 2*(Right-xHist[TissuePeak]) / 2.355
    if(Right == yHist.size):
        if(Left == -1):
            TissueScale = DefaultTissueScale
        else:
            LeftSlope = yHist[Left+1]-yHist[Left] / (xHist[Left+1]-xHist[Left])
            Left = (yHist[TissuePeak] / 2 - yHist[Left]) / LeftSlope + xHist[Left]
            TissueScale = 2*(xHist[TissuePeak]-Left) / 2.355

    #solve for mixing parameter
    Mix = yHist[BGPeak] * (BGScale * (2*numpy.pi)**0.5)

    #flatten kernel-smoothed histogram and corresponding x values for optimization
    xHist = xHist.flatten()
    yHist = yHist.flatten()

    #define gaussian mixture model
    def GaussianMixture(x, mu1, mu2, sigma1, sigma2, p):
        rv1 = norm(loc=mu1, scale=sigma1)
        rv2 = norm(loc=mu2, scale=sigma2)
        return p * rv1.pdf(x) + (1-p) * rv2.pdf(x)

    #define gaussian mixture model residuals
    def GaussianResiduals(Parameters, y, x):
        mu1, mu2, sigma1, sigma2, p = Parameters
        yhat = GaussianMixture(x, mu1, mu2, sigma1, sigma2, p)
        return sum((y-yhat) ** 2)

    #fit Gaussian mixture model and unpack results
    Parameters = fmin_slsqp(GaussianResiduals, [BGPeak, TissuePeak, BGScale, TissueScale, Mix],
                            args=(yHist, xHist), bounds=[(0, 255),(0, 255),(numpy.spacing(1), 10), (numpy.spacing(1), 50),(0,1)])
    muBackground = Parameters[0]
    muTissue = Parameters[1]
    sigmaBackground = Parameters[2]
    sigmaTissue = Parameters[3]
    p = Parameters[4]
    plt.plot(xHist, yHist)
    plt.plot(xHist, GaussianMixture(xHist, muBackground, muTissue, sigmaBackground, sigmaTissue, p))

    #create mask based on Gaussian mixture model
    Background = norm(loc=muBackground, scale=sigmaBackground)
    Tissue = norm(loc=muTissue, scale=sigmaTissue)
    pBackground = p * Background.pdf(xHist)
    pTissue = (1-p) * Tissue.pdf(xHist)

    #identify maximum likelihood threshold
    Difference = pTissue-pBackground
    Candidates = numpy.nonzero(Difference >= 0)[0]
    Filtered = numpy.nonzero(xHist[Candidates] > muBackground)
    ML = xHist[Candidates[Filtered[0]][0]]

    #identify limits for tissue model (MinProb, 1-MinProb)
    Endpoints = numpy.asarray(Tissue.interval(1-MinProb/2))

    #invert threshold and tissue mean
    ML = 255 - ML
    muTissue = 255 - muTissue
    Endpoints = numpy.sort(255-Endpoints)

    #generate mask
    Mask = (I <= ML) & (I >= Endpoints[0]) & (I <= Endpoints[1])
    Mask = Mask.astype(numpy.uint8)
    
    return(Mask)