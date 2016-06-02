from ctk_cli import CLIArgumentParser
import histomicstk as htk
import numpy as np
import json
import scipy as sp
import skimage.io
import skimage.measure

import logging
logging.basicConfig()

stainColorMap = {
    'hematoxylin': [0.65, 0.70, 0.29],
    'eosin':       [0.07, 0.99, 0.11],
    'dab':         [0.27, 0.57, 0.78],
    'null':        [0.0, 0.0, 0.0]
}


def main(args):

    #
    # Read Input Image
    #
    print('>> Reading input image')

    imInput = skimage.io.imread(args.inputImageFile)[:, :, :3]

    #
    # Perform color normalization
    #
    print('>> Performing color normalization')

    # transform input image to LAB color space
    imInputLAB = htk.RudermanLABFwd(imInput)

    # compute mean and stddev of input in LAB color space
    Mu = np.zeros(3)
    Sigma = np.zeros(3)

    for i in range(3):
        Mu[i] = imInputLAB[:, :, i].mean()
        Sigma[i] = (imInputLAB[:, :, i] - Mu[i]).std()

    # perform reinhard normalization
    imNmzd = htk.ReinhardNorm(imInput, Mu, Sigma)

    #
    # Perform color deconvolution
    #
    print('>> Performing color deconvolution')

    stainColor_1 = stainColorMap[args.stain_1]
    stainColor_2 = stainColorMap[args.stain_2]
    stainColor_3 = stainColorMap[args.stain_3]

    W = np.array([stainColor_1, stainColor_2, stainColor_3]).T

    imDeconvolved = htk.ColorDeconvolution(imNmzd, W)

    imNucleiStain = imDeconvolved.Stains[:, :, 0].astype(np.float)

    #
    # Perform nuclei segmentation
    #
    print('>> Performing nuclei segmentation')

    # segment foreground
    imFgndMask = sp.ndimage.morphology.binary_fill_holes(
        imNucleiStain < args.foreground_threshold)

    # run adaptive multi-scale LoG filter
    imLog = htk.cLoG(imNucleiStain, imFgndMask,
                     SigmaMin=args.min_radius * np.sqrt(2),
                     SigmaMax=args.max_radius * np.sqrt(2))

    imNucleiSegMask, Seeds, Max = htk.MaxClustering(
        imLog, imFgndMask, args.local_max_search_radius)

    # filter out small objects
    imNucleiSegMask = htk.FilterLabel(
        imNucleiSegMask, Lower=args.min_nucleus_area).astype(np.int)

    #
    # Generate annotations
    #
    objProps = skimage.measure.regionprops(imNucleiSegMask)

    print 'Number of nuclei = ', len(objProps)

    # create basic schema
    annotation = {
        "name":          "Nuclei",
        "description":   "Nuclei bounding boxes from a segmentation algorithm",
        "attributes": {
            "algorithm": {
                "color_normalization": "ReinhardNorm",
                "color_deconvolution": "ColorDeconvolution",
                "nuclei_segmentation": ["cLOG",
                                        "MaxClustering",
                                        "FilterLabel"]
            }
        },
        "elements": []
    }

    # add each nucleus as an element into the annotation schema
    for i in range(len(objProps)):

        c = [objProps[i].centroid[1], objProps[i].centroid[0], 0]
        width = objProps[i].bbox[3] - objProps[i].bbox[1] + 1
        height = objProps[i].bbox[2] - objProps[i].bbox[0] + 1

        cur_bbox = {
            "type":        "rectangle",
            "center":      c,
            "width":       width,
            "height":      height,
        }

        annotation["elements"].append(cur_bbox)

    #
    # Save output segmentation mask
    #
    print('>> Outputting nuclei segmentation mask')

    skimage.io.imsave(args.outputNucleiMaskFile, imNucleiSegMask)

    #
    # Save output annotation
    #
    print('>> Outputting nuclei annotation')

    with open(args.outputNucleiAnnotationFile, 'w') as annotationFile:
        json.dump(annotation, annotationFile, indent=2, sort_keys=False)


if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())
