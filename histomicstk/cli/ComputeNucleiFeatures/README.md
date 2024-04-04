HistomicsTK ComputeNucleiFeatures Application
=============================================

#### Overview:

Computes the following features that can be used for nuclei
classification:

* Morphometry (Size and shape) features
* Fourier shape descriptor features
* Intensity features from the nucleus and cytoplasm channels
* Gradient features from the nucleus and cytoplasm channels
* Haralick texture features for nucleus and cytoplasm channels

Each of the aforementioned groups of features can be toggled on/off
as needed.

The output of this application is a csv file wherein each row contains
the features of one of the nuclei.

#### Usage:

```
 ComputeNucleiFeatures.py [-h] [-V] [--xml] [--analysis_mag <double>]
                            [--analysis_roi <region>]
                            [--analysis_tile_size <double>]
                            [--cyto_width <integer>]
                            [--cytoplasm <boolean>]
                            [--foreground_threshold <double>]
                            [--fsd <boolean>] [--fsd_bnd_pts <integer>]
                            [--fsd_freq_bins <integer>]
                            [--gradient <boolean>] [--haralick <boolean>]
                            [--intensity <boolean>]
                            [--local_max_search_radius <double>]
                            [--max_radius <double>]
                            [--min_fgnd_frac <double>]
                            [--min_nucleus_area <double>]
                            [--min_radius <double>]
                            [--morphometry <boolean>]
                            [--nuclei_annotation_format {bbox,boundary}]
                            [--num_glcm_levels <integer>]
                            [--output_annotation_file <file>]
                            [--reference_mu_lab <double-vector>]
                            [--reference_std_lab <double-vector>]
                            [--scheduler <string>]
                            [--stain_1 {hematoxylin,eosin,dab,custom}]
                            [--stain_1_vector <double-vector>]
                            [--stain_2 {hematoxylin,eosin,dab,custom}]
                            [--stain_2_vector <double-vector>]
                            [--stain_3 {hematoxylin,eosin,dab,null,custom}]
                            [--stain_3_vector <double-vector>]
                            inputImageFile outputFile

positional arguments:
  inputImageFile        Input image (type: image)
  outputFile            Output CSV file (type: file) (file-extensions:
                        ['.csv'])

optional arguments:
  -h, --help            show this help message and exit
  -V, --version         show program's version number and exit
  --xml                 Produce xml description of command line arguments
  --analysis_mag <double>
                        The magnification at which the analysis should be
                        performed. (default: 20.0)
  --analysis_roi <region>
                        Region of interest within which the analysis should be
                        done. Must be a four element vector in the format
                        "left, top, width, height" in the space of the base
                        layer. Default value of "-1, -1, -1, -1" indicates
                        that the whole image should be processed. (default:
                        [-1.0, -1.0, -1.0, -1.0])
  --analysis_tile_size <double>
                        Tile size for blockwise analysis (default: 4096.0)
  --cyto_width <integer>
                        Width of ring-like neighborhood region around each
                        nucleus to be considered as cytoplasm (default: 8)
  --cytoplasm <boolean>
                        Compute Intensity and Gradient features from the
                        cytoplasm channel (default: True)
  --foreground_threshold <double>
                        Intensity value to use as threshold to segment
                        foreground in nuclear stain image (default: 60.0)
  --fsd <boolean>       Compute Fourier Shape Descriptor Features (default:
                        True)
  --fsd_bnd_pts <integer>
                        Number of boundary points for computing FSD features
                        (default: 128)
  --fsd_freq_bins <integer>
                        Number of frequency bins for calculating FSD features
                        (default: 6)
  --gradient <boolean>  Compute Gradient/Edge Features (default: True)
  --haralick <boolean>  Compute Haralick Texture Features (default: True)
  --intensity <boolean>
                        Compute Intensity Features (default: True)
  --local_max_search_radius <double>
                        Local max search radius used for detection seed points
                        in nuclei (default: 10.0)
  --max_radius <double>
                        Maximum nuclear radius (used to set max sigma of the
                        multiscale LoG filter) (default: 30.0)
  --min_fgnd_frac <double>
                        The minimum amount of foreground that must be present
                        in a tile for it to be analyzed (default: 0.5)
  --min_nucleus_area <double>
                        Minimum area that each nucleus should have (default:
                        80.0)
  --min_radius <double>
                        Minimum nuclear radius (used to set min sigma of the
                        multiscale LoG filter) (default: 20.0)
  --morphometry <boolean>
                        Compute Morphometry (Size and Shape) Features
                        (default: True)
  --nuclei_annotation_format {bbox,boundary}
                        Format of the output nuclei annotations (default:
                        boundary)
  --num_glcm_levels <integer>
                        Number of GLCM intensity levels (used to compute
                        haralick features) (default: 32)
  --output_annotation_file <file>
                        Output nuclei annotation file (file-extensions:
                        ['.anot'])
  --reference_mu_lab <double-vector>
                        Mean of reference image in LAB color space for
                        Reinhard color normalization (default: [8.63234435,
                        -0.11501964, 0.03868433])
  --reference_std_lab <double-vector>
                        Standard deviation of reference image in LAB color
                        space for Reinhard color normalization (default:
                        [0.57506023, 0.10403329, 0.01364062])
  --scheduler <string>
                        Address of a dask scheduler in the format
                        '127.0.0.1:8786'.  Not passing this parameter sets up a
                        dask cluster on the local machine.  'multiprocessing'
                        uses Python multiprocessing.  'multithreading' uses
                        Python multiprocessing in threaded mode.
  --stain_1 {hematoxylin,eosin,dab,custom}
                        Name of stain-1 (default: hematoxylin)
  --stain_1_vector <double-vector>
                        Custom value for stain-1 (default: [-1.0, -1.0, -1.0])
  --stain_2 {hematoxylin,eosin,dab,custom}
                        Name of stain-2 (default: eosin)
  --stain_2_vector <double-vector>
                        Custom value for stain-2 (default: [-1.0, -1.0, -1.0])
  --stain_3 {hematoxylin,eosin,dab,null,custom}
                        Name of stain-3 (default: null)
  --stain_3_vector <double-vector>
                        Custom value for stain-3 (default: [-1.0, -1.0, -1.0])

Title: Computes Nuclei Features

Description: Computes features for nuclei classification

Author(s): Deepak Roy Chittajallu (Kitware), Sanghoon Lee (Emory University)

License: Apache 2.0

Acknowledgements: This work is part of the HistomicsTK project.
```
