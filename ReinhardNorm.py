import numpy

def ReinhardNorm(I, TargetMu, TargetSigma):
'''
Performs Reinhard color normalization to transform the color characteristics of an image to 
a desired standard. The standard is defined by the mean and standard deviations of
the target image in LAB color space defined by Ruderman. The input image is converted to 
Ruderman's LAB space, the LAB channels are each centered and scaled to zero-mean unit 
variance, and then rescaled and shifted to match the target image statistics.
*Inputs:
	I (rgbimage) - an RGB image of type unsigned char.
	TargetMu - a 3-element list containing the means of the target image channels in LAB 
				color space.
	TargetSigma - a 3-element list containing the standard deviations of the target image
					channels in LAB color space.
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
LAB = RudermanLAB(I)

#center and scale to zero-mean and unit variance
Mu = LAB.sum(axis=0).sum(axis=0)
LAB[:,:,0] = LAB[:,:,0] - Mu[0]
LAB[:,:,1] = LAB[:,:,1] - Mu[1]
LAB[:,:,2] = LAB[:,:,2] - Mu[2]
Sigma = (LAB*LAB).sum(axis=0).sum(axis=0) / (m*n-1)
LAB[:,:,0] = LAB[:,:,0] / Sigma[0]
LAB[:,:,1] = LAB[:,:,1] / Sigma[1]
LAB[:,:,2] = LAB[:,:,2] / Sigma[2]

#rescale and recenter to match target statistics
LAB[:,:,0] = LAB[:,:,0] * TargetSigma[0] + TargetMu[0]
LAB[:,:,1] = LAB[:,:,1] * TargetSigma[1] + TargetMu[1]
LAB[:,:,2] = LAB[:,:,2] * TargetSigma[2] + TargetMu[2]

#convert back to RGB colorspace
Normalized = RudermanLABInv(LAB)

return(Normalized)
