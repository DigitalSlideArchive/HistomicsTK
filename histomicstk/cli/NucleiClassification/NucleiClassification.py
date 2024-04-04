import colorsys
import json
import os
from pathlib import Path

import dask
import dask.dataframe as dd
import large_image
import numpy as np
import pandas as pd

import histomicstk.segmentation.label as htk_seg_label
import histomicstk.segmentation.nuclear as htk_nuclear

try:
    import joblib
except ImportError:
    # Versions of scikit-learn before 0.21 had joblib internally
    from sklearn.externals import joblib

import logging

import histomicstk
from histomicstk.cli import utils as cli_utils
from histomicstk.cli.utils import CLIArgumentParser

logging.basicConfig(level=logging.CRITICAL)


def set_reference_values(args):
    """
    Set reference values and configuration parameters for feature extraction.

    Args:
    ----
        args (dict): Configuration parameters for feature extraction.

    Returns:
    -------
        dict: Updated configuration parameters with reference values set.

    """
    args.reference_mu_lab = [8.63234435, -0.11501964, 0.03868433]
    args.reference_std_lab = [0.57506023, 0.10403329, 0.01364062]
    args.foreground_threshold = 60
    args.min_radius = 6
    args.max_radius = 20
    args.min_nucleus_area = 80
    args.local_max_search_radius = 10
    args.nuclei_annotation_format = 'boundary'
    args.stain_1 = 'hematoxylin'
    args.stain_1_vector = [-1.0, -1.0, -1.0]
    args.stain_2 = 'eosin'
    args.stain_2_vector = [-1.0, -1.0, -1.0]
    args.stain_3 = 'null'
    args.stain_3_vector = [-1.0, -1.0, -1.0]
    args.ignore_border_nuclei = False
    args.cyto_width = 8
    args.cytoplasm_features = True
    args.fsd_bnd_pts = 128
    args.fsd_features = True
    args.fsd_freq_bins = 6
    args.gradient_features = True
    args.haralick_features = True
    args.morphometry_features = True
    args.intensity_features = True
    args.gradient_features = True
    args.fsd_features = True
    args.num_glcm_levels = 32
    args.min_fgnd_frac = .25
    args.analysis_roi = None
    return args


def gen_distinct_rgb_colors(n, seed=None):
    """
    Generates N visually distinct RGB colors

    Parameters
    ----------
    n : int
        Number of distinct RGB colors to output

    seed: int or array_like, optional
        Seed for the random number generator. See documentation of
        `numpy.random.seed` for more details

    Returns
    -------
    colors_list : list
        A list of n RGB colors

    References
    ----------
    .. [#] http://martin.ankerl.com/2009/12/09/how-to-create-random-colors-programmatically/  # noqa

    """
    np.random.seed(seed)
    h = np.random.random()
    np.random.seed(None)

    golden_ratio_conjugate = (np.sqrt(5) - 1) / 2.0

    color_list = [colorsys.hsv_to_rgb((h + i * golden_ratio_conjugate) % 1,
                                      1.0, 1.0)
                  for i in range(n)]

    return color_list


def process_feature_and_annotation(args):
    """
    Process nuclei feature extraction and annotation from an input image.

    Args:
    ----
        args (dict): Configuration parameters for feature extraction.

    Returns:
    -------
        tuple: A tuple containing nuclei annotations (list) and feature data (Dask DataFrame).

    """
    print('>> Generating features and annotation')

    #
    # Set arguments required for nuclei feature extraction
    #
    args = set_reference_values(args)
    tile_overlap = (args.max_radius + 1) * 4
    it_kwargs = {'tile_overlap': {'x': tile_overlap, 'y': tile_overlap}}

    #
    # Read Input Image
    #
    print('\n>> Reading input image ... \n')

    ts = large_image.getTileSource(args.inputImageFile)

    ts_metadata = ts.getMetadata()

    print(json.dumps(ts_metadata, indent=2))

    src_mu_lab = None
    src_sigma_lab = None

    #
    # Detect and compute nuclei features in parallel using Dask
    #
    print('\n>> Detecting nuclei and computing features ...\n')

    tile_result_list = []

    for tile in ts.tileIterator(**it_kwargs):

        # detect nuclei
        cur_result = dask.delayed(htk_nuclear.detect_tile_nuclei)(
            tile,
            args,
            src_mu_lab, src_sigma_lab,
            return_fdata=True,
        )

        # append result to list
        tile_result_list.append(cur_result)

    tile_result_list = dask.delayed(tile_result_list).compute()

    nuclei_annot_list = [annot
                         for annot_list, fdata in tile_result_list
                         for annot in annot_list]

    # remove overlapping nuclei
    nuclei_annot_list = htk_seg_label.remove_overlap_nuclei(
        nuclei_annot_list, args.nuclei_annotation_format)

    nuclei_fdata = pd.DataFrame()

    if len(nuclei_annot_list) > 0:

        nuclei_fdata = pd.concat([
            fdata
            for annot_list, fdata in tile_result_list if fdata is not None],
            ignore_index=True,
        )
    # Fill any instances with NaN as zero
    df = pd.DataFrame(nuclei_fdata).fillna(0)
    return nuclei_annot_list, dd.from_pandas(df, npartitions=1)


def read_feature_file(args):
    """
    Read nuclei feature data from a specified file.

    Args:
    ----
        args (dict): Configuration parameters including the input feature file path.

    Returns:
    -------
        dask.dataframe.DataFrame: A Dask DataFrame containing the nuclei feature data.

    """
    fname, feature_file_format = os.path.splitext(args.inputNucleiFeatureFile)

    if feature_file_format == '.csv':

        ddf = dd.read_csv(args.inputNucleiFeatureFile)

    elif feature_file_format == '.h5':

        ddf = dd.read_hdf(args.inputNucleiFeatureFile, 'Features')

    else:
        msg = 'Extension of output feature file must be .csv or .h5'
        raise ValueError(msg)

    # Fill any instances with NaN as zero
    return ddf.fillna(0)


def main(args):

    print('\n>> CLI Parameters ...\n')

    print(args)

    #
    # Initiate Dask client
    #
    print('\n>> Creating Dask client ...\n')

    c = cli_utils.create_dask_client(args)

    print(c)

    #
    # read model file
    #
    print('\n>> Loading classification model ...\n')
    clf_model = joblib.load(args.inputModelFile)

    if args.inputNucleiFeatureFile and args.inputNucleiAnnotationFile:

        # read feature file
        print('\n>> Loading nuclei feature file ...\n')

        ddf = read_feature_file(args)

        if len(ddf.columns) != clf_model.n_features_in_:

            msg = (
                'The number of features of the classification model '
                'and the input feature file do not match.'
            )
            raise ValueError(msg)

        #
        # read nuclei annotation file
        #
        print('\n>> Loading nuclei annotation file ...\n')

        with open(args.inputNucleiAnnotationFile) as f:

            annotation_data = json.load(f)
            nuclei_annot_list = annotation_data.get(
                'elements', annotation_data.get(
                    'annotation', {}).get('elements'))

        if len(nuclei_annot_list) != len(ddf.index):

            msg = (
                'The number of nuclei in the feature file and the '
                'annotation file do not match'
            )
            raise ValueError(msg)
    else:
        nuclei_annot_list, ddf = process_feature_and_annotation(args)

    #
    # Perform nuclei classification
    #
    print('\n>> Performing nuclei classification using Dask ...\n')

    def predict_nuclei_class_prob(df, clf_model):

        return pd.DataFrame(data=clf_model.predict_proba(df.values),
                            columns=clf_model.classes_)

    outfmt = pd.DataFrame(columns=clf_model.classes_, dtype=np.float64)

    df_class_prob = ddf.map_partitions(predict_nuclei_class_prob, clf_model,
                                       meta=outfmt).compute()

    pred_class = df_class_prob.idxmax(axis=1)

    #
    # Group nuclei annotations by class
    #
    print('\n>> Grouping nuclei annotations by class ...\n')

    num_classes = len(clf_model.classes_)

    nuclei_annot_by_class = {c: [] for c in clf_model.classes_}

    class_color_map = dict(zip(clf_model.classes_,
                               gen_distinct_rgb_colors(num_classes, seed=1)))

    for i in range(len(nuclei_annot_list)):

        cur_class = pred_class.iloc[i]

        cur_anot = nuclei_annot_list[i]
        cur_anot['lineColor'] = 'rgb(%s)' % ','.join(
            [str(int(round(col * 255))) for col in class_color_map[cur_class]])
        nuclei_annot_by_class[cur_class].append(cur_anot)

    #
    # Write annotation file
    #
    print('\n>> Writing classified nuclei annotation file ...\n')

    annot_fname = os.path.splitext(
        os.path.basename(args.outputNucleiAnnotationFile))[0]

    annotation = []
    for c in clf_model.classes_:

        annotation.append({
            'name': 'Class ' + str(c) + ' ' + annot_fname,
            'elements': nuclei_annot_by_class[c],
        })
    annotation.append({
        'attributes': {
            'params': vars(args),
            'cli': Path(__file__).stem,
            'version': histomicstk.__version__,

        },
    })
    with open(args.outputNucleiAnnotationFile, 'w') as annotation_file:
        json.dump(annotation, annotation_file, separators=(',', ':'), sort_keys=False)


if __name__ == '__main__':
    main(CLIArgumentParser().parse_args())
