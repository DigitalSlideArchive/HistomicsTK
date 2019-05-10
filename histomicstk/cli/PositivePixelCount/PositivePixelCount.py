import os

from histomicstk.cli.utils import CLIArgumentParser
import large_image
import numpy as np
import skimage.io

import histomicstk.segmentation.positive_pixel_count as ppc

from histomicstk.cli import utils


def main(args):
    utils.create_dask_client(args)
    ts = large_image.getTileSource(args.inputImageFile)
    make_label_image = getattr(args, 'outputLabelImage', None) is not None
    region = utils.get_region_dict(
        args.region,
        *(args.maxRegionSize, ts) if make_label_image else ()
    ).get('region')
    ppc_params = ppc.Parameters(
        **{k: getattr(args, k) for k in ppc.Parameters._fields}
    )
    results = ppc.count_slide(
        args.inputImageFile, ppc_params, region,
        args.tile_grouping, make_label_image,
    )
    if make_label_image:
        stats, label_image = results
        # Colorize label image.  Colors from the "coolwarm" color map
        color_map = np.empty((4, 3), dtype=np.uint8)
        color_map[ppc.Labels.NEGATIVE] = 255
        color_map[ppc.Labels.WEAK] = 60, 78, 194
        color_map[ppc.Labels.PLAIN] = 221, 220, 220
        color_map[ppc.Labels.STRONG] = 180, 4, 38
        # Cleverly index color_map
        label_image = color_map[label_image]
        try:
            skimage.io.imsave(args.outputLabelImage, label_image)
        except ValueError:
            # This is likely caused by an unknown extension, so try again
            altname = args.outputLabelImage + '.png'
            skimage.io.imsave(altname, label_image)
            os.rename(altname, args.outputLabelImage)
    else:
        stats, = results
    with open(args.returnParameterFile, 'w') as f:
        for k, v in zip(stats._fields, stats):
            f.write('{} = {}\n'.format(k, v))


if __name__ == '__main__':
    main(CLIArgumentParser().parse_args())
