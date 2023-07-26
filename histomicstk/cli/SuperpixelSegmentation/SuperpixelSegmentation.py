import json
import math
import os
from pathlib import Path

import large_image
import numpy
import pyvips
import scipy
import skimage

import histomicstk
from histomicstk.cli import utils
from histomicstk.cli.utils import CLIArgumentParser


def createSuperPixels(opts):  # noqa
    """
    Besides the inputs described by the specification, this also can take
    opts.callback, which is a function that takes (step_name: str,
    step_count: int, step_total: int).
    """
    averageSize = opts.superpixelSize ** 2
    overlap = opts.superpixelSize * 4 * 2 if opts.overlap else 0
    tileSize = opts.tileSize + overlap

    print('>> Reading input image')
    print(opts.inputImageFile)

    ts = large_image.open(opts.inputImageFile)
    meta = ts.getMetadata()
    found = 0
    strips = []
    bboxes = []
    bboxesUser = []
    tiparams = {}
    tiparams = utils.get_region_dict(opts.roi, None, ts)

    scale = 1
    if opts.magnification:
        tiparams['scale'] = {'magnification': opts.magnification}

    print('>> Generating superpixels')
    if opts.slic_zero:
        print('>> Using SLIC Zero for segmentation')
    for tile in ts.tileIterator(
        format=large_image.constants.TILE_FORMAT_NUMPY,
        tile_size=dict(width=tileSize, height=tileSize),
        tile_overlap=dict(x=overlap, y=overlap),
        **tiparams,
    ):
        if hasattr(opts, 'callback'):
            opts.callback('tiles', tile['tile_position']['position'],
                          tile['iterator_range']['position'])
        print('%d/%d (%d x %d) - %d' % (
            tile['tile_position']['position'], tile['iterator_range']['position'],
            tile['width'], tile['height'],
            found))
        if meta['magnification'] and tile['magnification']:
            scale = meta['magnification'] / tile['magnification']
        x0 = tiparams.get('region', {}).get('left', 0)
        y0 = tiparams.get('region', {}).get('top', 0)
        # tx0 = tile['x'] - x0
        # ty0 = tile['y'] - y0
        tx0 = int((tile['gx'] - x0) / scale)
        ty0 = int((tile['gy'] - y0) / scale)
        img = tile['tile']
        n_pixels = tile['width'] * tile['height']
        mask = None
        if overlap:
            mask = numpy.ones(img.shape[:2])
            for y, simg in strips:
                if (y < ty0 + tile['height'] and y + simg.height > ty0 and simg.width > tx0):
                    suby = max(0, y - ty0)
                    subimg = simg.crop(
                        tx0,
                        max(0, ty0 - y),
                        min(tile['width'], simg.width),
                        min(tile['height'], simg.height - max(0, ty0 - y)))
                    # Our mask is true when a pixel has not been set
                    submask = numpy.ndarray(
                        buffer=subimg[3].write_to_memory(),
                        dtype=numpy.uint8,
                        shape=[subimg.height, subimg.width]) == 0
                    mask[suby:suby + submask.shape[0], :submask.shape[1]] *= submask
            n_pixels = numpy.count_nonzero(mask)
        n_segments = math.ceil(n_pixels / averageSize)
        segments = skimage.segmentation.slic(
            img,
            n_segments=n_segments,
            slic_zero=bool(opts.slic_zero),
            compactness=opts.compactness,
            sigma=opts.sigma,
            start_label=0,
            enforce_connectivity=True,
            mask=mask,
        )
        # We now have an array that is the same size as the image
        maxValue = numpy.max(segments) + 1
        if overlap:
            # Keep any segment that is at all in the non-overlap region
            core = segments[
                :tile['height'] - tile['tile_overlap']['bottom'],
                :tile['width'] - tile['tile_overlap']['right']]
            coremask = mask[
                :tile['height'] - tile['tile_overlap']['bottom'],
                :tile['width'] - tile['tile_overlap']['right']]
            core[numpy.where(coremask != 1)] = -1
            usedIndices = numpy.unique(core)
            usedIndices = numpy.delete(usedIndices, numpy.where(usedIndices < 0))
            usedLut = [-1] * maxValue
            for idx, used in enumerate(usedIndices):
                if used >= 0:
                    usedLut[used] = idx
            usedLut = numpy.array(usedLut, dtype=int)
            print('reduced from %d to %d' % (maxValue, len(usedIndices)))
            maxValue = len(usedIndices)
            segments = usedLut[segments]
            mask *= (segments != -1)
        if str(opts.bounding).lower() not in {'', 'none'}:
            regions = skimage.measure.regionprops(1 + segments)
            for pidx, props in enumerate(regions):
                by0, bx0, by1, bx1 = props.bbox
                bboxes.append((
                    ((bx0 + bx1) / 2 + tx0) * scale + x0,
                    ((by0 + by1) / 2 + ty0) * scale + y0,
                    (bx1 - bx0) * scale,
                    (by1 - by0) * scale))
                bboxesUser.extend([
                    (bx0 + tx0) * scale + x0,
                    (by0 + ty0) * scale + y0,
                    (bx1 + tx0) * scale + x0,
                    (by1 + ty0) * scale + y0,
                ])
        if opts.boundaries:
            segments *= 2
            maxValue *= 2
            edges = (scipy.ndimage.sobel(segments, axis=0) != 0) | (
                scipy.ndimage.sobel(segments, axis=1) != 0)
            edges[0, :] = True
            edges[-1, :] = True
            edges[:, 0] = True
            edges[:, -1] = True
            segments += edges
        segments += found
        found += int(maxValue)
        if mask is None:
            data = numpy.dstack((
                (segments % 256).astype(int),
                (segments / 256).astype(int) % 256,
                (segments / 65536).astype(int) % 256)).astype('B')
        else:
            data = numpy.dstack((
                (segments % 256).astype(int),
                (segments / 256).astype(int) % 256,
                (segments / 65536).astype(int) % 256,
                mask * 255)).astype('B')
        # For overlay, suppose we make any value whose centroid is in the
        # overlap region transparent.  Then, use vips to overlap the
        # images rather than inserting them.
        vimg = pyvips.Image.new_from_memory(
            numpy.ascontiguousarray(data).data,
            data.shape[1], data.shape[0], data.shape[2],
            large_image.constants.dtypeToGValue[data.dtype.char])
        vimg = vimg.copy(interpretation=pyvips.Interpretation.RGB)
        vimgTemp = pyvips.Image.new_temp_file('%s.v')
        vimg.write(vimgTemp)
        vimg = vimgTemp
        x = tx0
        ty = tile['tile_position']['region_y']
        while len(strips) <= ty:
            strips.append(None)
        if strips[ty] is None:
            strip = pyvips.Image.black(
                tiparams.get('region', {}).get('width', meta['sizeX']),
                vimg.height, bands=vimg.bands)
            strip = strip.copy(interpretation=pyvips.Interpretation.RGB)
            strips[ty] = [ty0, strip]
        strips[ty][1] = strips[ty][1].composite([vimg], pyvips.BlendMode.OVER, x=int(x), y=0)
        if hasattr(opts, 'callback'):
            opts.callback('tiles', tile['tile_position']['position'] + 1,
                          tile['iterator_range']['position'])
    if hasattr(opts, 'callback'):
        opts.callback('file', 0, 2 if opts.outputAnnotationFile else 1)
    print('>> Found %d superpixels' % found)
    if found > 256 ** 3:
        print('Too many superpixels')
    img = pyvips.Image.black(
        tiparams.get('region', {}).get('width', meta['sizeX']) / scale,
        tiparams.get('region', {}).get('height', meta['sizeY']) / scale,
        bands=strips[0][1].bands)
    img = img.copy(interpretation=pyvips.Interpretation.RGB)
    for stripidx in range(len(strips)):
        img = img.composite(
            [strips[stripidx][1]], pyvips.BlendMode.OVER, x=0, y=int(strips[stripidx][0]))
    # Discard alpha band, if any.
    img = img[:3]
    # Add program run parameters to the image description and list the
    # superpixel count
    img.set_type(
        pyvips.GValue.gstr_type, 'image-description',
        json.dumps(dict(
            {k: v for k, v in vars(opts).items() if k != 'callback'}, indexCount=found)))
    img.write_to_file(
        opts.outputImageFile, tile=True, tile_width=256, tile_height=256, pyramid=True,
        region_shrink=pyvips.RegionShrink.NEAREST,
        # We'd prefer max, but to do so we need to compute max of the
        # superpixel, not the faux-color it is mapped to.
        # region_shrink=pyvips.RegionShrink.MAX,
        bigtiff=True, compression='lzw', predictor='horizontal')

    if hasattr(opts, 'callback'):
        opts.callback('file', 1, 2 if opts.outputAnnotationFile else 1)
    if opts.outputAnnotationFile:
        print('>> Generating annotation file')
        categories = [
            {
                'label': opts.default_category_label,
                'fillColor': opts.default_fillColor,
                'strokeColor': opts.default_strokeColor,
            },
        ]
        annotation_name = os.path.splitext(os.path.basename(opts.outputAnnotationFile))[0]
        region_dict = utils.get_region_dict(opts.roi, None, ts)
        annotation = {
            'name': annotation_name,
            'elements': [{
                'type': 'pixelmap',
                'girderId': 'outputImageFile',
                'transform': {
                    'xoffset': region_dict.get('region', {}).get('left', 0) / scale,
                    'yoffset': region_dict.get('region', {}).get('top', 0) / scale,
                    'matrix': [[scale, 0], [0, scale]],
                },
                'values': [0] * (found // (2 if opts.boundaries else 1)),
                'categories': categories,
                'boundaries': opts.boundaries,
            }],
            'attributes': {
                'params': vars(opts),
                'cli': Path(__file__).stem,
                'version': histomicstk.__version__,
            },
        }
        if len(bboxes) and str(opts.bounding).lower() != 'separate':
            annotation['elements'][0]['user'] = {'bbox': bboxesUser}
        if len(bboxes) and str(opts.bounding).lower() != 'internal':
            bboxannotation = {
                'name': '%s bounding boxes' % os.path.splitext(
                    os.path.basename(opts.outputAnnotationFile))[0],
                'elements': [{
                    'type': 'rectangle',
                    'center': [bcx, bcy, 0],
                    'width': bw,
                    'height': bh,
                    'rotation': 0,
                    'label': {'value': 'Region %d' % bidx},
                    'fillColor': 'rgba(0,0,0,0)',
                    'lineColor': opts.default_strokeColor,
                } for bidx, (bcx, bcy, bw, bh) in enumerate(bboxes)],
                'attributes': {
                    'params': vars(opts),
                    'cli': Path(__file__).stem,
                    'version': histomicstk.__version__,
                },
            }
            annotation = [annotation, bboxannotation]
        with open(opts.outputAnnotationFile, 'w') as annotation_file:
            json.dump(annotation, annotation_file, separators=(',', ':'), sort_keys=False)
        if hasattr(opts, 'callback'):
            opts.callback('file', 2, 2)


if __name__ == '__main__':
    createSuperPixels(CLIArgumentParser().parse_args())
