HistomicsTK ComputeNucleiFeatures Application
=============================================

#### Overview:

Computes the following features that can be used for nuclei 
classification:

* Morphometry (Size and shape) features
* Fourier shape descriptor features
* Intensity features from the nucleus and cytoplasm channels
* Gradient features from the nucleus and cytoplasm channels

Each of the aformentioned groups of features can be toggled on/off
as needed.

The output of this application is a HDF5 file containing the features.

#### Usage:

```
ComputeNucleiFeatures.py [-h] [-V] [--xml] [--cyto_width <integer>]
                         [--cytoplasm <boolean>]
                         [--foreground_threshold <double>]
                         [--fsd <boolean>] [--fsd_bnd_pts <integer>]
                         [--fsd_freq_bins <integer>]
                         [--gradient <boolean>] [--intensity <boolean>]
                         [--local_max_search_radius <double>]
                         [--max_radius <double>]
                         [--min_nucleus_area <double>]
                         [--min_radius <double>]
                         [--size_shape <boolean>]
                         [--stain_1 {hematoxylin,eosin,dab}]
                         [--stain_2 {hematoxylin,eosin,dab}]
                         [--stain_3 {hematoxylin,eosin,dab,null}]
                         inputImageFile outputFile

positional arguments:
  inputImageFile        Input image (type: image)
  outputFile            Output HDF5 file (type: file) (file-extensions:
                        ['.h5'])

optional arguments:
  -h, --help            show this help message and exit
  -V, --version         show program's version number and exit
  --xml                 Produce xml description of command line arguments
  --cyto_width <integer>
                        Width of ring-like neighborghood region around each
                        nucleus to be considered as cytoplasm (default: 8)
  --cytoplasm <boolean>
                        Compute Intensity and Gradient features from the
                        cytoplasm channel (default: True)
  --foreground_threshold <double>
                        Intensity value to use as threshold to segment
                        foreground in nuclear stain image (default: 160.0)
  --fsd <boolean>       Compute Fourier Shape Descriptor Features (default:
                        True)
  --fsd_bnd_pts <integer>
                        Number of boundary points for computing FSD features
                        (default: 128)
  --fsd_freq_bins <integer>
                        Number of frequency bins for calculating FSD features
                        (default: 6)
  --gradient <boolean>  Compute Gradient/Edge Features (default: True)
  --intensity <boolean>
                        Compute Intensity Features (default: True)
  --local_max_search_radius <double>
                        Local max search radius used for detection seed points
                        in nuclei (default: 10.0)
  --max_radius <double>
                        Maximum nuclear radius (used to set max sigma of the
                        multiscale LoG filter) (default: 7.0)
  --min_nucleus_area <double>
                        Minimum area that each nucleus should have (default:
                        80.0)
  --min_radius <double>
                        Minimum nuclear radius (used to set min sigma of the
                        multiscale LoG filter) (default: 4.0)
  --size_shape <boolean>
                        Compute Size and shape Features (default: True)
  --stain_1 {hematoxylin,eosin,dab}
                        Name of stain-1 (default: hematoxylin)
  --stain_2 {hematoxylin,eosin,dab}
                        Name of stain-2 (default: eosin)
  --stain_3 {hematoxylin,eosin,dab,null}
                        Name of stain-3 (default: null)

Title: Computes Nuclei Classification Features

Description: Computes features for nuclei classification

Author(s): Sanghoon Lee (Emory University), Deepak Roy Chittajallu (Kitware)

License: Apache 2.0

Acknowledgements: This work is part of the HistomicsTK project.
```