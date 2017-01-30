import histomicstk.preprocessing.color_conversion as htk_ccvt
import histomicstk.preprocessing.color_normalization as htk_cnorm
import histomicstk.preprocessing.color_deconvolution as htk_cdeconv
import histomicstk.filters.shape as htk_shape_filters
import histomicstk.segmentation as htk_seg

import numpy as np
import json
import scipy as sp
import skimage.io
import skimage.measure

from ctk_cli import CLIArgumentParser

import logging
logging.basicConfig()


stain_color_map = {
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

    im_input = skimage.io.imread(args.inputImageFile)[:, :, :3]

    #
    # Perform color normalization
    #
    print('>> Performing color normalization')

    # compute mean and stddev of input in LAB color space
    mu, sigma = htk_ccvt.lab_mean_std(im_input)

    # perform reinhard normalization
    im_nmzd = htk_cnorm.reinhard(im_input, mu, sigma)

    #
    # Perform color deconvolution
    #
    print('>> Performing color deconvolution')

    stain_color_1 = stain_color_map[args.stain_1]
    stain_color_2 = stain_color_map[args.stain_2]
    stain_color_3 = stain_color_map[args.stain_3]

    w = np.array([stain_color_1, stain_color_2, stain_color_3]).T

    im_stains = htk_cdeconv.color_deconvolution(im_nmzd, w).Stains

    im_nuclei_stain = im_stains[:, :, 0].astype(np.float)

    #
    # Perform nuclei segmentation
    #
    print('>> Performing nuclei segmentation')

    # segment foreground
    im_fgnd_mask = sp.ndimage.morphology.binary_fill_holes(
        im_nuclei_stain < args.foreground_threshold)

    # run adaptive multi-scale LoG filter
    im_log = htk_shape_filters.clog(im_nuclei_stain, im_fgnd_mask,
                                    sigma_min=args.min_radius * np.sqrt(2),
                                    sigma_max=args.max_radius * np.sqrt(2))

    im_nuclei_seg_mask, seeds, max = htk_seg.nuclear.max_clustering(
        im_log, im_fgnd_mask, args.local_max_search_radius)

    # filter out small objects
    im_nuclei_seg_mask = htk_seg.label.area_open(
        im_nuclei_seg_mask, args.min_nucleus_area).astype(np.int)

    #
    # Generate annotations
    #
    obj_props = skimage.measure.regionprops(im_nuclei_seg_mask)

    print 'Number of nuclei = ', len(obj_props)

    # create basic schema
    annotation = {
        "name":          "Nuclei",
        "description":   "Nuclei bounding boxes from a segmentation algorithm",
        "attributes": {
            "algorithm": {
                "color_normalization": "reinhard",
                "color_deconvolution": "ColorDeconvolution",
                "nuclei_segmentation": ["cLOG",
                                        "MaxClustering",
                                        "FilterLabel"]
            }
        },
        "elements": []
    }

    # add each nucleus as an element into the annotation schema
    for i in range(len(obj_props)):

        c = [obj_props[i].centroid[1], obj_props[i].centroid[0], 0]
        width = obj_props[i].bbox[3] - obj_props[i].bbox[1] + 1
        height = obj_props[i].bbox[2] - obj_props[i].bbox[0] + 1

        cur_bbox = {
            "type":        "rectangle",
            "center":      c,
            "width":       width,
            "height":      height,
            "rotation":    0,
            "fillColor":   "rgba(255, 255, 255, 0)",
            "lineWidth":   2,
            "lineColor":   "rgb(34, 139, 34)"
        }

        annotation["elements"].append(cur_bbox)

    #
    # Save output segmentation mask
    #
    print('>> Outputting nuclei segmentation mask')

    skimage.io.imsave(args.outputNucleiMaskFile, im_nuclei_seg_mask)

    #
    # Save output annotation
    #
    print('>> Outputting nuclei annotation')

    with open(args.outputNucleiAnnotationFile, 'w') as annotation_file:
        json.dump(annotation, annotation_file, indent=2, sort_keys=False)


if __name__ == "__main__":
    main(CLIArgumentParser().parse_args())
