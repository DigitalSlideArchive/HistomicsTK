# Tests TilingSchedule.py. Generates a tiling schedule, and then sequentially
# reads in all tiles, shrinks them significantly, and recomposes the original
# image. The re-composed image should appear coherent.

import matplotlib.pyplot as plt
import numpy
import openslide
import scipy

# Define globals for the style checker
# TODO: Ensure these are actually provided
TilingSchedule = TilingSchedule  # noqa

# resizing factor for display
Shrink = 0.1

# Define input whole-slide image, tile size, and tiling magnification for a
# magnification that is contained in the WSI file
File = './CMU-1.svs'
Magnification = 5
Tile = 2048

# get tiling schedule for slide
Schedule = TilingSchedule(File, Magnification, Tile)

# open image
Slide = openslide.OpenSlide(File)

# calculate tilesize of output tiles for visualization
Tshrink = numpy.floor(Tile * Shrink)

# re-compose image by grabbing each tile in the schedule, and resizing it into
# a list
Image = numpy.zeros((Tshrink * Schedule.X.shape[0],
                     Tshrink * Schedule.X.shape[1], 4), dtype=numpy.uint8)
for i in range(Schedule.X.shape[0]):
    for j in range(Schedule.X.shape[1]):

        # read region
        Region = Slide.read_region((int(Schedule.X[i, j]),
                                    int(Schedule.Y[i, j])),
                                   Schedule.Level,
                                   (Schedule.Tout, Schedule.Tout))

        # resize if desired magnification is not provided by the file
        if Schedule.Factor != 1.0:
            Region = scipy.misc.imresize(Region, Schedule.Factor)

        # resize region for visualization purposes
        Region = scipy.misc.imresize(Region, Shrink)

        # copy to output image
        Image[i * Tshrink:(i + 1) * Tshrink,
              j * Tshrink:(j + 1) * Tshrink, :] = Region

        # update console
        print("Processing tile (%s, %s) of (%s, %s)\n." %
              (i, j, Schedule.X.shape[0] - 1, Schedule.X.shape[1] - 1))

# view output - check if coherent
plt.imshow(Image)
