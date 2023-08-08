import colorsys
import json
import os
import time
from pathlib import Path

import numpy as np

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

    #
    # Detect and compute nuclei features in parallel using Dask
    #
    print('\n>> Detecting nuclei and computing features ...\n')

    start_time = time.time()

    tile_result_list = []

    for tile in ts.tileIterator(**it_kwargs):

        tile_position = tile['tile_position']['position']

        if is_wsi and tile_fgnd_frac_list[tile_position] <= args.min_fgnd_frac:
            continue

        # detect nuclei
        cur_result = dask.delayed(compute_tile_nuclei_features)(
            args.inputImageFile,
            tile_position,
            args, it_kwargs,
            src_mu_lab, src_sigma_lab
        )

        # append result to list
        tile_result_list.append(cur_result)

    tile_result_list = dask.delayed(tile_result_list).compute()

    nuclei_annot_list = [annot
                         for annot_list, fdata in tile_result_list
                         for annot in annot_list]

    nuclei_fdata = pd.DataFrame()

    if len(nuclei_annot_list) > 0:

        nuclei_fdata = pd.concat([
            fdata
            for annot_list, fdata in tile_result_list if fdata is not None],
            ignore_index=True
        )
    return None, None

def read_feature_file(args):
    import dask.dataframe as dd

    fname, feature_file_format = os.path.splitext(args.inputNucleiFeatureFile)

    if feature_file_format == '.csv':

        ddf = dd.read_csv(args.inputNucleiFeatureFile)

    elif feature_file_format == '.h5':

        ddf = dd.read_hdf(args.inputNucleiFeatureFile, 'Features')

    else:
        raise ValueError('Extension of output feature file must be .csv or .h5')

    return ddf


def main(args):
    import pandas as pd

    print('\n>> CLI Parameters ...\n')

    print(args)

    #
    # Initiate Dask client
    #
    print('\n>> Creating Dask client ...\n')

    c = cli_utils.create_dask_client(args)

    print(c)

    #
    # Check if input feature files are available
    #
    if not os.path.isfile(args.inputImageFile) and  not os.path.isfile(args.inputModelFile):
        process_feature_annotation()

    #
    # read model file
    #
    print('\n>> Loading classification model ...\n')

    clf_model = joblib.load(args.inputModelFile)

    if args.inputImageFile and args.inputModelFile:
        # read feature file
        #
        print('\n>> Loading nuclei feature file ...\n')

        ddf = read_feature_file(args)

        if len(ddf.columns) != clf_model.n_features_in_:

            raise ValueError('The number of features of the classification model '
                            'and the input feature file do not match.')

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

            raise ValueError('The number of nuclei in the feature file and the '
                            'annotation file do not match')
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
            'name': annot_fname + '-nuclei-class-' + str(c),
            'elements': nuclei_annot_by_class[c]
        })
    annotation.append({
        'attributes': {
            'params': vars(args),
            'cli': Path(__file__).stem,
            'version': histomicstk.__version__

        }
    })
    with open(args.outputNucleiAnnotationFile, 'w') as annotation_file:
        json.dump(annotation, annotation_file, separators=(',', ':'), sort_keys=False)


if __name__ == '__main__':
    main(CLIArgumentParser().parse_args())
