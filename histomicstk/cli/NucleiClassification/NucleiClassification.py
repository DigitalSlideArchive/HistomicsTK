import colorsys
import json
import os
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


def check_args(args):

    if not os.path.isfile(args.inputImageFile):
        raise OSError('Input image file does not exist.')

    if not os.path.isfile(args.inputModelFile):
        raise OSError('Input model file does not exist.')


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
    # read model file
    #
    print('\n>> Loading classification model ...\n')

    clf_model = joblib.load(args.inputModelFile)

    #
    # read feature file
    #
    print('\n>> Loading nuclei feature file ...\n')

    ddf = read_feature_file(args)

    if len(ddf.columns) != clf_model.n_features_:

        raise ValueError('The number of features of the classification model '
                         'and the input feature file do not match.')

    #
    # read nuclei annotation file
    #
    print('\n>> Loading nuclei annotation file ...\n')

    with open(args.inputNucleiAnnotationFile) as f:

        nuclei_annot_list = json.load(f)['elements']

    if len(nuclei_annot_list) != len(ddf.index):

        raise ValueError('The number of nuclei in the feature file and the '
                         'annotation file do not match')

    #
    # Perform nuclei classification
    #
    print('\n>> Performing nuclei classification using Dask ...\n')

    def predict_nuclei_class_prob(df, clf_model):

        return pd.DataFrame(data=clf_model.predict_proba(df.as_matrix()),
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
