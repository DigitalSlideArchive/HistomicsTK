import os
import pyvips
import numpy as np
from imageio import imread
import matplotlib.pylab as plt

# % ===========================================================================

width = 10000
height = 10000

MOSAIC_PATH = '/home/mtageld/tmp/bigim.tif'

TILE_DIR = '/home/mtageld/tmp/'
TILE_PATHS = [
    os.path.join(TILE_DIR, j) for j in
    os.listdir(TILE_DIR) if j.endswith('.png')]

# % ===========================================================================

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

# this makes a 8-bit, mono image of height x width pixels, each pixel zero
im = pyvips.Image.black(width, height, bands=3)
im.rawsave(MOSAIC_PATH)

# load exiting tiles
# im = pyvips.Image.rawload(location, width, height, 3)

# % ===========================================================================

# # The following first loads tiles into memory!!
# # This may be OK, but also keep in mind that there is a way to directly
# # add the tiles to the VIPS image by provided the tile path
#
# tile = tile1
# xpos = 1000
# ypos = 2000
#
# ydim, xdim, channels = tile.shape
# linear = tile.reshape(ydim * xdim * channels)
# vi = pyvips.Image.new_from_memory(
#     linear.data, xdim, ydim, 3, dtype_to_format[str(linear.dtype)])
#
# im = im.draw_image(vi, xpos, ypos)
#

# % ===========================================================================

# Option 2: This insead directly read srom file
# THIS is the way to go for: 1. efficiency; 2. you can just
# determine the NUMBER (not size) of tiles you want per row and columns
# and let vips expand as it adds images.
# Something like here: https://github.com/libvips/pyvips/issues/109

tilepath = TILE_PATHS[0]
xpos = 9500  # <- expand to accomodate!!
ypos = 1000

tile = pyvips.Image.new_from_file(tilepath, access="sequential")
im = im.insert(tile, xpos, ypos, expand=True)

# % ===========================================================================

print("Saving created VIPS image to " + MOSAIC_PATH)
im.vipssave(MOSAIC_PATH)

# % ===========================================================================
# % ===========================================================================

