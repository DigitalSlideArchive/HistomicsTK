import json
import math
import os
from pathlib import Path

import large_image
import large_image_source_zarr
import numpy as np
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
    bboxes = []
    bboxesUser = []
    tiparams = {}
    tiparams = utils.get_region_dict(opts.roi, None, ts)

    scale = 1
    if opts.magnification:
        tiparams['scale'] = {'magnification': opts.magnification}

    sink = large_image_source_zarr.new()
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
        tx0 = int((tile['gx'] - x0) / scale)
        ty0 = int((tile['gy'] - y0) / scale)
        img = tile['tile']
        if img.shape[2] in {2, 4}:
            img = img[:, :, :-1]
        n_pixels = tile['width'] * tile['height']
        mask = None
        if overlap:
            if sink.sizeX:
                mask, _ = sink.getRegion(
                    format=large_image.constants.TILE_FORMAT_NUMPY,
                    region=dict(
                        left=tx0, top=ty0, width=tile['width'], height=tile['height']))
                mask = (mask[:, :, -1] == 0)
            else:
                mask = np.ones(img.shape[:2], dtype=bool)
            if mask.shape[0] < img.shape[0] or mask.shape[1] < img.shape[1]:
                mask2 = np.ones(img.shape[:2], dtype=bool)
                mask2[:mask.shape[0], :mask.shape[1]] = mask
                mask = mask2
            n_pixels = np.count_nonzero(mask)
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
        maxValue = np.max(segments) + 1
        if overlap:
            # Keep any segment that is at all in the non-overlap region
            core = segments[
                :tile['height'] - tile['tile_overlap']['bottom'],
                :tile['width'] - tile['tile_overlap']['right']]
            coremask = mask[
                :tile['height'] - tile['tile_overlap']['bottom'],
                :tile['width'] - tile['tile_overlap']['right']]
            core[np.where(coremask != 1)] = -1
            usedIndices = np.unique(core)
            usedIndices = np.delete(usedIndices, np.where(usedIndices < 0))
            usedLut = [-1] * maxValue
            for idx, used in enumerate(usedIndices):
                if used >= 0:
                    usedLut[used] = idx
            usedLut = np.array(usedLut, dtype=int)
            print('reduced from %d to %d' % (maxValue, len(usedIndices)))
            maxValue = len(usedIndices)
            segments = usedLut[segments]
            mask *= (segments != -1)
        if str(opts.bounding).lower() not in {'', 'none'}:
            regions = skimage.measure.regionprops(1 + segments)
            for _pidx, props in enumerate(regions):
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
            data = segments.astype(np.uint32)[..., np.newaxis]
        else:
            data = np.dstack((segments.astype(np.uint32), mask)).astype(np.uint32)
        sink.addTile(data, tx0, ty0, mask=mask)
        if hasattr(opts, 'callback'):
            opts.callback('tiles', tile['tile_position']['position'] + 1,
                          tile['iterator_range']['position'])
    if hasattr(opts, 'callback'):
        opts.callback('file', 0, 2 if opts.outputAnnotationFile else 1)
    print('>> Found %d superpixels' % found)
    if found > 256 ** 3:
        print('Too many superpixels')
    sink2 = large_image_source_zarr.new()
    for y in range(0, sink.sizeY, 4096):
        for x in range(0, sink.sizeX, 4096):
            tile = sink.getRegion(
                format=large_image.constants.TILE_FORMAT_NUMPY,
                region=dict(
                    left=x, top=y, width=min(sink.sizeX - x, 4096),
                    height=min(sink.sizeY - y, 4096)))[0][:, :, 0]
            tile = np.dstack((
                (tile % 256).astype(int),
                ((tile // 256) % 256).astype(int),
                ((tile // 65536) % 256).astype(int))).astype('B')
            sink2.addTile(tile, x, y)
    # Add program run parameters to the image description and list the
    # superpixel count
    # img.set_type(
    #     pyvips.GValue.gstr_type, 'image-description',
    #     json.dumps(dict(
    #         {k: v for k, v in vars(opts).items() if k != 'callback'}, indexCount=found)))
    sink2.write(opts.outputImageFile, lossy=False)

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
                'params': {k: v for k, v in vars(opts).items() if not callable(v)},
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
                    'params': {k: v for k, v in vars(opts).items() if not callable(v)},
                    'cli': Path(__file__).stem,
                    'version': histomicstk.__version__,
                },
            }
            annotation = [annotation, bboxannotation]
        with open(opts.outputAnnotationFile, 'w') as annotation_file:
            try:
                json.dump(annotation, annotation_file, separators=(',', ':'), sort_keys=False)
            except Exception:
                print('Failed to serialize annotation')
                print(repr(annotation))
                raise
        if hasattr(opts, 'callback'):
            opts.callback('file', 2, 2)


if __name__ == '__main__':
    createSuperPixels(CLIArgumentParser().parse_args())
