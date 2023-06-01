"""
Created on Mon Sep 23 21:17:43 2019.

@author: mtageld
"""
import numpy as np
from PIL import Image
from skimage.color import rgb2gray

from histomicstk.annotations_and_masks.annotation_and_mask_utils import \
    get_image_from_htk_response
from histomicstk.annotations_and_masks.masks_to_annotations_handler import (
    get_annotation_documents_from_contours, get_contours_from_mask)
from histomicstk.features.compute_haralick_features import \
    compute_haralick_features
from histomicstk.features.compute_intensity_features import \
    compute_intensity_features
from histomicstk.preprocessing.color_conversion import lab_mean_std
from histomicstk.preprocessing.color_normalization import reinhard
from histomicstk.saliency.tissue_detection import (
    _deconv_color, get_slide_thumbnail,
    get_tissue_boundary_annotation_documents, get_tissue_mask)
from histomicstk.utils.general_utils import Base_HTK_Class

Image.MAX_IMAGE_PIXELS = None


class CD_single_tissue_piece:
    """Detect cellular regions in a single tissue piece (internal)."""

    def __init__(self, cd, tissue_mask, monitorPrefix=''):
        """Detect cellularity in one tissue piece (Internal).

        Arguments
        ----------
        cd : object
            Cellularity_detector_superpixels instance
        tissue_mask : np array
            (mxn) mask of the tissue piece at cd.MAG magnification
        monitorPrefix : str
            Text to prepend to printed statements

        """
        self.cd = cd
        self.tissue_mask = 0 + tissue_mask
        self.monitorPrefix = monitorPrefix

    def run(self):
        """Get cellularity and optionally visualize on DSA."""
        # get cellularity
        self.restrict_mask_to_single_tissue_piece()
        self.cd._print2('%s: set_tissue_rgb()' % self.monitorPrefix)
        self.set_tissue_rgb()
        self.cd._print2('%s: set_superpixel_mask()' % self.monitorPrefix)
        self.set_superpixel_mask()
        self.cd._print2('%s: set_superpixel_features()' % self.monitorPrefix)
        self.set_superpixel_features()
        self.cd._print2('%s: set_superpixel_assignment()' % self.monitorPrefix)
        self.set_superpixel_assignment()
        self.cd._print2('%s: assign_cellularity_scores()' % self.monitorPrefix)
        self.assign_cellularity_scores()

        # visualize
        if self.cd.visualize_spixels or self.cd.visualize_contiguous:
            self.assign_colors_to_spixel_clusters()

        if self.cd.visualize_spixels:
            self.cd._print2(
                '%s: visualize_individual_superpixels()' % self.monitorPrefix)
            self.visualize_individual_superpixels()

        if self.cd.visualize_contiguous:
            self.cd._print2(
                '%s: visualize_contiguous_superpixels()' % self.monitorPrefix)
            self.visualize_contiguous_superpixels()

    def restrict_mask_to_single_tissue_piece(self):
        """Only keep relevant part of slide mask."""
        # find coordinates at scan magnification
        tloc = np.argwhere(self.tissue_mask)
        F = self.cd.slide_info['F_tissue']
        self.ymin, self.xmin = (int(j) for j in np.min(tloc, axis=0) * F)
        self.ymax, self.xmax = (int(j) for j in np.max(tloc, axis=0) * F)
        self.tissue_mask = self.tissue_mask[
            int(self.ymin / F): int(self.ymax / F),
            int(self.xmin / F): int(self.xmax / F)]

    def set_tissue_rgb(self):
        """Load RGB from server for single tissue piece."""
        # load RGB for this tissue piece at saliency magnification
        getStr = '/item/%s/tiles/region?left=%d&right=%d&top=%d&bottom=%d&encoding=PNG' % (
            self.cd.slide_id, self.xmin, self.xmax, self.ymin, self.ymax
        ) + '&magnification=%d' % self.cd.MAG
        resp = self.cd.gc.get(getStr, jsonResp=False)
        self.tissue_rgb = get_image_from_htk_response(resp)

        # color normalization if desired
        if 'main' in self.cd.cnorm_params.keys():
            self.tissue_rgb = np.uint8(reinhard(
                im_src=self.tissue_rgb,
                target_mu=self.cd.cnorm_params['main']['mu'],
                target_sigma=self.cd.cnorm_params['main']['sigma']))

    def set_superpixel_mask(self):
        """Use Simple Linear Iterative Clustering (SLIC) to get superpixels."""
        from skimage.segmentation import slic
        from skimage.transform import resize

        # Get superpixel size and number
        spixel_size = self.cd.spixel_size_baseMag * (
            self.cd.MAG / self.cd.slide_info['magnification'])
        n_spixels = int(
            self.tissue_rgb.shape[0] * self.tissue_rgb.shape[1] / spixel_size)

        # get superpixel mask
        # optionally use grayscale instead of RGB -- seems more robust to
        # color variations and sometimes gives better results
        if self.cd.use_grayscale:
            try:
                self.spixel_mask = slic(
                    rgb2gray(self.tissue_rgb), n_segments=n_spixels,
                    compactness=self.cd.compactness, channel_axis=None)
            except Exception:
                self.spixel_mask = slic(
                    rgb2gray(self.tissue_rgb), n_segments=n_spixels,
                    compactness=self.cd.compactness)
        else:
            self.spixel_mask = slic(
                self.tissue_rgb, n_segments=n_spixels,
                compactness=self.cd.compactness)

        # restrict to tissue mask
        tmask = resize(
            self.tissue_mask, output_shape=self.spixel_mask.shape,
            order=0, preserve_range=True, anti_aliasing=False)
        self.spixel_mask[tmask == 0] = 0

    def set_superpixel_features(self):
        """Get superpixel features."""
        from pandas import concat
        from skimage.measure import regionprops

        assert (self.cd.use_intensity or self.cd.use_texture)

        # Possibly deconvolvve to get hematoxylin channel (cellular areas)
        # hematoxylin channel return shows MINIMA so we invert
        if self.cd.deconvolve:
            Stains, channel = _deconv_color(self.tissue_rgb)
            tissue_htx = 255 - Stains[..., channel]
        else:
            tissue_htx = rgb2gray(self.tissue_rgb)

        # calculate features from superpixels
        rprops = regionprops(self.spixel_mask)
        fdata_list = []
        if self.cd.use_intensity:
            fdata_list.append(compute_intensity_features(
                im_label=self.spixel_mask, im_intensity=tissue_htx,
                rprops=rprops))
        if self.cd.use_texture:
            fdata_list.append(compute_haralick_features(
                im_label=self.spixel_mask, im_intensity=tissue_htx,
                rprops=rprops))
        self.fdata = concat(fdata_list, axis=1)

        if self.cd.keep_feats is not None:
            self.fdata = self.fdata.loc[:, self.cd.keep_feats]

        # Index is corresponding pixel value in the superpixel mask
        # IMPORTANT: this assumes that regionprops output is sorted by unique
        # pixel values in label mask, which it is by default
        self.fdata.index = set(np.unique(self.spixel_mask)) - {0, }

    def set_superpixel_assignment(self):
        """Fit gaussian mixture model to features and get assignment."""
        from sklearn.mixture import GaussianMixture

        mmodel = GaussianMixture(n_components=self.cd.n_gaussian_components)
        self.spixel_labels = mmodel.fit_predict(self.fdata.values) + 1

    def assign_cellularity_scores(self):
        """Assign cellularity scores to spixel clusters."""
        assert self.cd.use_intensity, 'We need intensity to rank cellularity.'
        assert self.cd.deconvolve, \
            'We must use hematoxyling channel to rank by cellularity.'

        self.cluster_props = {}

        self.fdata.loc[:, 'cluster'] = self.spixel_labels
        for clid in np.unique(self.spixel_labels):
            self.cluster_props[clid] = {
                'cellularity': int(np.median(self.fdata.loc[
                    self.fdata.loc[:, 'cluster'] == clid,
                    'Intensity.Median']) / 255 * 100),
            }

    def assign_colors_to_spixel_clusters(self):
        """Assign RGB color string to cellularity clusters."""
        # normalize values by given value, else by max for each tissue piece
        if self.cd.max_cellularity is not None:
            max_cellularity = self.cd.max_cellularity
        else:
            max_cellularity = max(
                j['cellularity'] for _, j in self.cluster_props.items()
            )

        # Assign rgb string
        for clid in np.unique(self.spixel_labels):
            cellularity = min(
                self.cluster_props[clid]['cellularity'], max_cellularity)
            rgb = self.cd.cMap(int(cellularity / max_cellularity * 255))[:-1]
            rgb = [int(255 * j) for j in rgb]
            self.cluster_props[clid]['color'] = 'rgb(%d,%d,%d)' % tuple(rgb)

    def visualize_individual_superpixels(self):
        """Visualize individual spixels, color-coded by cellularity."""
        # Define GTCodes dataframe
        from pandas import DataFrame

        GTCodes_df = DataFrame(columns=['group', 'GT_code', 'color'])
        for spval, sp in self.fdata.iterrows():
            spstr = 'spixel-%d_cellularity-%d' % (
                spval, self.cluster_props[sp['cluster']]['cellularity'])
            GTCodes_df.loc[spstr, 'group'] = spstr
            GTCodes_df.loc[spstr, 'GT_code'] = spval
            GTCodes_df.loc[spstr, 'color'] = \
                self.cluster_props[sp['cluster']]['color']

        # get contours df
        contours_df = get_contours_from_mask(
            MASK=self.spixel_mask, GTCodes_df=GTCodes_df,
            get_roi_contour=False, MIN_SIZE=0, MAX_SIZE=None,
            verbose=self.cd.verbose == 3, monitorPrefix=self.monitorPrefix)
        contours_df.loc[:, 'group'] = [
            j.split('_')[-1] for j in contours_df.loc[:, 'group']]

        # get annotation docs
        annprops = {
            'F': (self.ymax - self.ymin) / self.tissue_rgb.shape[0],
            'X_OFFSET': self.xmin,
            'Y_OFFSET': self.ymin,
            'opacity': self.cd.opacity,
            'lineWidth': self.cd.lineWidth,
        }
        annotation_docs = get_annotation_documents_from_contours(
            contours_df.copy(), docnamePrefix='spixel', annprops=annprops,
            annots_per_doc=1000, separate_docs_by_group=True,
            verbose=self.cd.verbose == 3, monitorPrefix=self.monitorPrefix)
        for didx, doc in enumerate(annotation_docs):
            self.cd._print2('%s: Posting doc %d of %d' % (
                self.monitorPrefix, didx + 1, len(annotation_docs)))
            _ = self.cd.gc.post(
                '/annotation?itemId=' + self.cd.slide_id, json=doc)

    def visualize_contiguous_superpixels(self):
        """Visualize contiguous spixels, color-coded by cellularity."""
        from pandas import DataFrame

        # get cellularity cluster membership mask
        cellularity_mask = np.zeros(self.spixel_mask.shape)
        for spval, sp in self.fdata.iterrows():
            cellularity_mask[self.spixel_mask == spval] = sp['cluster']

        # Define GTCodes dataframe
        GTCodes_df = DataFrame(columns=['group', 'GT_code', 'color'])
        for spval, cp in self.cluster_props.items():
            spstr = 'cellularity-%d' % (cp['cellularity'])
            GTCodes_df.loc[spstr, 'group'] = spstr
            GTCodes_df.loc[spstr, 'GT_code'] = spval
            GTCodes_df.loc[spstr, 'color'] = cp['color']

        # get contours df
        contours_df = get_contours_from_mask(
            MASK=cellularity_mask, GTCodes_df=GTCodes_df,
            get_roi_contour=False, MIN_SIZE=0, MAX_SIZE=None,
            verbose=self.cd.verbose == 3, monitorPrefix=self.monitorPrefix)

        # get annotation docs
        annprops = {
            'F': (self.ymax - self.ymin) / self.tissue_rgb.shape[0],
            'X_OFFSET': self.xmin,
            'Y_OFFSET': self.ymin,
            'opacity': self.cd.opacity_contig,
            'lineWidth': self.cd.lineWidth,
        }
        annotation_docs = get_annotation_documents_from_contours(
            contours_df.copy(), docnamePrefix='contig', annprops=annprops,
            annots_per_doc=1000, separate_docs_by_group=True,
            verbose=self.cd.verbose == 3, monitorPrefix=self.monitorPrefix)
        for didx, doc in enumerate(annotation_docs):
            self.cd._print2('%s: Posting doc %d of %d' % (
                self.monitorPrefix, didx + 1, len(annotation_docs)))
            _ = self.cd.gc.post(
                '/annotation?itemId=' + self.cd.slide_id, json=doc)


class Cellularity_detector_superpixels (Base_HTK_Class):
    """Detect cellular regions in a slides by classifying superpixels.

    This uses Simple Linear Iterative Clustering (SLIC) to get superpixels at
    a low slide magnification to detect cellular regions. The first step of
    this pipeline detects tissue regions (i.e. individual tissue pieces)
    using the get_tissue_mask method of the histomicstk.saliency module. Then,
    each tissue piece is processed separately for accuracy and disk space
    efficiency. It is important to keep in mind that this does NOT rely on a
    tile iterator, but loads the entire tissue region (but NOT the whole slide)
    in memory and passes it on to skimage.segmentation.slic method.

    Once superpixels are segmented, the image is deconvolved and features are
    extracted from the hematoxylin channel. Features include intensity and
    possibly also texture features. Then, a mixed component Gaussian mixture
    model is fit to the features, and median intensity is used to rank
    superpixel clusters by 'cellularity' (since we are working with the
    hematoxylin channel).

    Additional functionality includes contour extraction to get the final
    segmentation boundaries of cellular regions and to visualize them in DSA
    using one's preferred colormap.

    """

    def __init__(self, gc, slide_id, **kwargs):
        """Init Cellularity_Detector_Superpixels object.

        Arguments:
        -----------
        gc : object
            girder client object
        slide_id : str
            girder ID of slide
        verbose : int
            0 - Do not print to screen
            1 - Print only key messages
            2 - Print everything to screen
            3 - print everything including from inner functions
        monitorPrefix : str
            text to prepend to printed statements
        logging_savepath : str or None
            where to save run logs
        suppress_warnings : bool
            whether to suppress warnings
        cnorm_params : dict
            Reinhard color normalization parameters. Accepted keys: thumbnail
            and main (since thumbnail normalization is different from color
            normalization of tissue at target magnification. Each entry is a
            dict containing values for mu and sigma. This is either given
            here or can be set using self.set_color_normalization_values().
            May be left unset if you do not want to normalize.
        get_tissue_mask_kwargs : dict
            kwargs for the get_tissue_mask() method.
        MAG : float
            magnification at which to detect cellularity
        spixel_size_baseMag : int
            approximate superpixel size at base (scan) magnification
        compactness : float
            compactness parameter for the SLIC method. Higher values result
            in more regular superpixels while smaller values are more likely
            to respect tissue boundaries.
        deconvolve : bool
            Whether to deconvolve and use hematoxylin channel for feature
            extraction. Must be True to ranks spixel clusters by cellularity.
        use_grayscale : bool
            If True, grayscale image is used with SLIC. May be more robust to
            color variations from slide to slide and more efficient.
        use_intensity : bool
            Whether to extract intensity features from the hematoxylin channel.
            This must be True to rank spuerpixel clusters by cellularity.
        use_texture : bool
            Whether to extract Haralick texture features from Htx channel. May
            not necessarily improve results when used in conjunction with
            intensity features.
        keep_feats : list
            Name of intensity features to use. See
            histomicstk.features.compute_intensity_features.
            Using fewer informative features may result in better
            gaussian mixture modeling results.
        n_gaussian_components : int
            no of gaussian mixture model components
        max_cellularity : int
            Range [0, 100] or None. If None, normalize visualization RGB values
            for each tissue piece separately, else normalize by given number.
        opacity : float
            opacity of superpixel polygons when posted to DSA.
            0 (no opacity) is more efficient to render.
        opacity_contig : float
            opacity of contiguous region polygons when posted to DSA.
            0 (no opacity) is more efficient to render.
        lineWidth : float
            width of line when displaying superpixel boundaries.
        cMap : object
            matplotlib color map to use when visualizing cellularity
        visualize_tissue_boundary : bool
            whether to visualize result from tissue detection component
        visualize_spixels : bool
            whether to visualize superpixels, color-coded by cellularity
        visualize_contiguous : bool
            whether to visualize contiguous cellular regions

        """
        from matplotlib import cm

        default_attr = {

            # The following are already assigned defaults by Base_HTK_Class
            # 'verbose': 1,
            # 'monitorPrefix': "",
            # 'logging_savepath': None,
            # 'suppress_warnings': False,

            'cnorm_params': {},
            'MAG': 3.0,
            'get_tissue_mask_kwargs': {
                'deconvolve_first': True, 'n_thresholding_steps': 1,
                'sigma': 1.5, 'min_size': 500,
            },
            'spixel_size_baseMag': 256 * 256,
            'compactness': 0.1,
            'deconvolve': True,
            'use_grayscale': True,
            'use_intensity': True,
            'use_texture': False,
            # 'keep_feats': None,  # keep everything
            'keep_feats': [
                'Intensity.Mean', 'Intensity.Median',
                'Intensity.Std', 'Intensity.IQR',
                'Intensity.HistEntropy',
            ],
            'n_gaussian_components': 5,
            'max_cellularity': None,
            'opacity': 0,
            'opacity_contig': 0.3,
            'lineWidth': 3.0,
            'cMap': cm.seismic,
            'visualize_tissue_boundary': True,
            'visualize_spixels': True,
            'visualize_contiguous': True,
        }
        default_attr.update(kwargs)
        super().__init__(
            default_attr=default_attr)

        # set attribs
        self.gc = gc
        self.slide_id = slide_id

    def run(self):
        """Run cellularity detection and optionally visualize result.

        This runs the cellularity detection +/- visualization pipeline and
        returns a list of CD_single_tissue_piece objects. Each object has
        the following attributes

        tissue_mask : np array
            mask of where tissue is at target magnification
        ymin : int
            min y coordinate at base (scan) magnification
        xmin : int
            min x coordinate at base (scan) magnification
        ymax : int
            max y coordinate at base (scan) magnification
        xmax : int
            max x coordinate at base (scan) magnification
        spixel_mask : np array
            np array where each unique value represents one superpixel
        fdata : pandas DataFrame
            features extracted for each superpixel. Index corresponds to
            values in the spixel_mask. This includes a 'cluster' column
            indicatign which cluster this superpixel belongs to.
        cluster_props : dict
            properties of each superpixel cluster, including its assigned
            cellularity score.

        """
        if (len(self.cnorm_params) == 0) and (not self.suppress_warnings):
            input("""
                %s: WARNING!! Consider running set_color_normalization_values()
                first, using what='thumbnail' and/or what='main' before running
                this method. Continue anyway?""" % self.monitorPrefix)

        # get mask, each unique value is a single tissue piece
        self._print1(
            '%s: set_slide_info_and_get_tissue_mask()' % self.monitorPrefix)
        labeled = self.set_slide_info_and_get_tissue_mask()

        # Go through tissue pieces and do run sequence
        unique_tvals = list(set(np.unique(labeled)) - {0, })
        tissue_pieces = [None for _ in range(len(unique_tvals))]
        for idx, tval in enumerate(unique_tvals):
            monitorPrefix = '%s: Tissue piece %d of %d' % (
                self.monitorPrefix, idx + 1, len(unique_tvals))
            self._print1(monitorPrefix)
            tissue_pieces[idx] = CD_single_tissue_piece(
                self, tissue_mask=labeled == tval, monitorPrefix=monitorPrefix)
            tissue_pieces[idx].run()
            del tissue_pieces[idx].tissue_rgb  # too much space

        return tissue_pieces

    def set_color_normalization_values(
            self, mu=None, sigma=None, ref_image_path=None, what='main'):
        """Set color normalization values for thumbnail or main image."""
        assert (
            all(j is not None for j in (mu, sigma)) or ref_image_path is not None
        ), 'You must provide mu & sigma values or ref. image to get them.'

        assert what in ('thumbnail', 'main')

        if ref_image_path is not None:
            from imageio import imread

            ref_im = np.array(imread(ref_image_path, pilmode='RGB'))
            mu, sigma = lab_mean_std(ref_im)

        self.cnorm_params[what] = {'mu': mu, 'sigma': sigma}

    def set_slide_info_and_get_tissue_mask(self):
        """Set self.slide_info dict and self.labeled tissue mask."""
        # This is a persistent dict to store information about slide
        self.slide_info = self.gc.get('item/%s/tiles' % self.slide_id)

        # get tissue mask
        thumbnail_rgb = get_slide_thumbnail(self.gc, self.slide_id)

        # color normalization if desired
        if 'thumbnail' in self.cnorm_params.keys():
            thumbnail_rgb = np.uint8(reinhard(
                im_src=thumbnail_rgb,
                target_mu=self.cnorm_params['thumbnail']['mu'],
                target_sigma=self.cnorm_params['thumbnail']['sigma']))

        # get labeled tissue mask -- each unique value is one tissue piece
        labeled, _ = get_tissue_mask(
            thumbnail_rgb, **self.get_tissue_mask_kwargs)

        if len(np.unique(labeled)) < 2:
            raise ValueError('No tissue detected!')

        if self.visualize_tissue_boundary:
            annotation_docs = get_tissue_boundary_annotation_documents(
                self.gc, slide_id=self.slide_id, labeled=labeled)
            for doc in annotation_docs:
                _ = self.gc.post(
                    '/annotation?itemId=' + self.slide_id, json=doc)

        # Find size relative to WSI
        self.slide_info['F_tissue'] = self.slide_info[
            'sizeX'] / labeled.shape[1]

        return labeled
