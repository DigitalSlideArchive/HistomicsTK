import os
import pyvips
import numpy as np
from imageio import imread
import matplotlib.pylab as plt

# % ===========================================================================

MOSAIC_PATH = '/home/mtageld/tmp/'

TILE_DIR = '/home/mtageld/Desktop/cTME/data/tcga-nucleus/' \
           + 'nucleus_images_2019-04-14_20_52_15.831294/' \
           + 'Annotation_images_2019-04-14_20_52_15.831294/visualization/'
TILE_PATHS = [
    os.path.join(TILE_DIR, j) for j in
    os.listdir(TILE_DIR) if j.endswith('.png')]

# % ===========================================================================

TILES_PER_ROW = 10
TILES_PER_COLUMN = 20

# source: https://libvips.github.io/libvips/API/current/Examples.md.html
# source 2: https://libvips.github.io/libvips/API/current/Examples.md.html
# source 3: https://github.com/libvips/pyvips/issues/109
# source 4: https://github.com/libvips/libvips/issues/1254

# map np dtypes to vips
dtype_to_format = {
    'uint8': 'uchar',
    'int8': 'char',
    'uint16': 'ushort',
    'int16': 'short',
    'uint32': 'uint',
    'int32': 'int',
    'float32': 'float',
    'float64': 'double',
    'complex64': 'complex',
    'complex128': 'dpcomplex',
}

# map vips formats to np dtypes
format_to_dtype = {
    'uchar': np.uint8,
    'char': np.int8,
    'ushort': np.uint16,
    'short': np.int16,
    'uint': np.uint32,
    'int': np.int32,
    'float': np.float32,
    'double': np.float64,
    'complex': np.complex64,
    'dpcomplex': np.complex128,
}

# % ===========================================================================

# Tiles are read directly read from file
# THIS is the way to go for: 1. efficiency; 2. you can just
# determine the NUMBER (not size) of tiles you want per row and columns
# and let vips expand as it adds images.
# Something like here: https://github.com/libvips/pyvips/issues/109



n_tiles = len(TILE_PATHS)
n_mosaics = int(np.ceil(n_tiles / (TILES_PER_ROW * TILES_PER_COLUMN)))


tileidx = 0

for mosno in range(n_mosaics):

    # this makes a 8-bit, mono image (initializes as 1x1x3 matrix)
    im = pyvips.Image.black(1, 1, bands=3)

    for row in range(TILES_PER_COLUMN):

        row_im = pyvips.Image.black(1, 1, bands=3)

        for col in range(TILES_PER_ROW):

            if tileidx == n_tiles:
                break

            tilepath = TILE_PATHS[tileidx]
            print("Inserting tile %d of %d: %s" % (tileidx, n_tiles, tilepath))
            tileidx += 1

            tile = pyvips.Image.new_from_file(tilepath, access="sequential")
            row_im = row_im.insert(tile[:3], row_im.width, 0, expand=True)

        im = im.insert(row_im, 0, im.height, expand=True)

    savepath = os.path.join(MOSAIC_PATH, 'mosaic-%d.tiff' % (mosno + 1))
    print("Saving mosiac %d of %d to %s" % (mosno + 1, n_mosaics, savepath))
    im.tiffsave(
        savepath, tile=True, tile_width=256, tile_height=256, pyramid=True)

# % ===========================================================================
# % ===========================================================================