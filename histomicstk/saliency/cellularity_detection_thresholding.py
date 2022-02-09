#!/usr/bin/env python3
"""
Created on Tue Oct 22 02:37:52 2019.

@author: mtageld
"""
import numpy as np
from PIL import Image

from histomicstk.annotations_and_masks.annotation_and_mask_utils import \
    get_image_from_htk_response
from histomicstk.annotations_and_masks.masks_to_annotations_handler import (
    get_annotation_documents_from_contours, get_contours_from_mask)
from histomicstk.features.compute_intensity_features import \
    compute_intensity_features
from histomicstk.preprocessing.color_conversion import (lab_mean_std,
                                                        rgb_to_hsi, rgb_to_lab)
from histomicstk.preprocessing.color_deconvolution import (
    _reorder_stains, color_deconvolution_routine,
    rgb_separate_stains_macenko_pca)
from histomicstk.preprocessing.color_normalization import (
    deconvolution_based_normalization, reinhard)
from histomicstk.saliency.tissue_detection import (_get_largest_regions,
                                                   get_slide_thumbnail,
                                                   get_tissue_mask,
                                                   threshold_multichannel)
from histomicstk.utils.general_utils import Base_HTK_Class

Image.MAX_IMAGE_PIXELS = None


class CDT_single_tissue_piece:
    """Detect various regions in a single tissue piece (internal)."""

    def __init__(self, cdt, tissue_mask, monitorPrefix=""):
        """Detect whitespace, saliency, etc in one tissue piece (Internal).

        Arguments
        ----------
        cdt : object
            Cellularity_detector_thresholding instance
        tissue_mask : np array
            (m x n) mask of the tissue piece at cdt.MAG magnification
        monitorPrefix : str
            Text to prepend to printed statements

        """
        self.cdt = cdt
        self.tissue_mask = 0 + tissue_mask
        self.monitorPrefix = monitorPrefix

    def run(self):
        """Get cellularity and optionally visualize on DSA."""
        self.restrict_mask_to_single_tissue_piece()
        self.cdt._print2("%s: set_tissue_rgb()" % self.monitorPrefix)
        self.set_tissue_rgb()
        self.cdt._print2("%s: initialize_labeled_mask()" % self.monitorPrefix)
        self.initialize_labeled_mask()
        self.cdt._print2(
            "%s: assign_components_by_thresholding()" % self.monitorPrefix)
        self.assign_components_by_thresholding()
        self.cdt._print2(
            "%s: color_normalize_unspecified_components()"
            % self.monitorPrefix)
        self.color_normalize_unspecified_components()
        self.cdt._print2(
            "%s: find_potentially_cellular_regions()" % self.monitorPrefix)
        self.find_potentially_cellular_regions()
        self.cdt._print2(
            "%s: find_top_cellular_regions()" % self.monitorPrefix)
        self.find_top_cellular_regions()
        if self.cdt.visualize:
            self.cdt._print2("%s: visualize_results()" % self.monitorPrefix)
            self.visualize_results()

    def restrict_mask_to_single_tissue_piece(self):
        """Only keep relevant part of slide mask."""
        # find coordinates at scan magnification
        tloc = np.argwhere(self.tissue_mask)
        F = self.cdt.slide_info['F_tissue']
        self.ymin, self.xmin = [int(j) for j in np.min(tloc, axis=0) * F]
        self.ymax, self.xmax = [int(j) for j in np.max(tloc, axis=0) * F]
        self.tissue_mask = self.tissue_mask[
            int(self.ymin / F): int(self.ymax / F),
            int(self.xmin / F): int(self.xmax / F)]

    def set_tissue_rgb(self):
        """Load RGB from server for single tissue piece."""
        # load RGB for this tissue piece at saliency magnification
        getStr = "/item/%s/tiles/region?left=%d&right=%d&top=%d&bottom=%d&encoding=PNG" % (
            self.cdt.slide_id, self.xmin, self.xmax, self.ymin, self.ymax
            ) + "&magnification=%d" % self.cdt.MAG
        resp = self.cdt.gc.get(getStr, jsonResp=False)
        self.tissue_rgb = get_image_from_htk_response(resp)

    def initialize_labeled_mask(self):
        """Initialize labeled components mask."""
        from skimage.transform import resize

        # resize tissue mask to target mag
        self.labeled = resize(
            self.tissue_mask, output_shape=self.tissue_rgb.shape[:2],
            order=0, preserve_range=True, anti_aliasing=False)
        self.labeled[self.labeled > 0] = self.cdt.GTcodes.loc[
            'not_specified', 'GT_code']

    def assign_components_by_thresholding(self):
        """Get components by thresholding in HSI and LAB spaces."""
        # get HSI and LAB images
        self.cdt._print2(
            "%s: -- get HSI and LAB images ..." % self.monitorPrefix)
        tissue_hsi = rgb_to_hsi(self.tissue_rgb)
        tissue_lab = rgb_to_lab(self.tissue_rgb)

        # extract components using HSI/LAB thresholds

        hsi_components = self.cdt.hsi_thresholds.keys()
        lab_components = self.cdt.lab_thresholds.keys()

        for component in self.cdt.ordered_components:

            self.cdt._print2("%s: -- thresholding %s ..." % (
                self.monitorPrefix, component))

            if component in hsi_components:
                lab, _ = threshold_multichannel(
                    tissue_hsi,
                    channels=['hue', 'saturation', 'intensity'],
                    thresholds=self.cdt.hsi_thresholds[component],
                    just_threshold=False,
                    get_tissue_mask_kwargs=self.cdt.get_tissue_mask_kwargs2)
            elif component in lab_components:
                lab, _ = threshold_multichannel(
                    tissue_lab,
                    channels=['l', 'a', 'b'],
                    thresholds=self.cdt.lab_thresholds[component],
                    just_threshold=True,
                    get_tissue_mask_kwargs=self.cdt.get_tissue_mask_kwargs2)
            else:
                raise ValueError("Unknown component name.")

            lab[self.labeled == 0] = 0  # restrict to tissue mask
            self.labeled[lab > 0] = self.cdt.GTcodes.loc[component, 'GT_code']

        # This deals with holes in tissue
        self.labeled[self.labeled == 0] = self.cdt.GTcodes.loc[
            'outside_tissue', 'GT_code']

    def color_normalize_unspecified_components(self):
        """Color normalize "true" tissue components."""
        if self.cdt.color_normalization_method == 'reinhard':
            self.cdt._print2(
                "%s: -- reinhard normalization ..." % self.monitorPrefix)
            self.tissue_rgb = reinhard(
                self.tissue_rgb,
                target_mu=self.cdt.target_stats_reinhard['mu'],
                target_sigma=self.cdt.target_stats_reinhard['sigma'],
                mask_out=self.labeled != self.cdt.GTcodes
                    .loc["not_specified", "GT_code"])

        elif self.cdt.color_normalization_method == 'macenko_pca':
            self.cdt._print2(
                "%s: -- macenko normalization ..." % self.monitorPrefix)
            self.tissue_rgb = deconvolution_based_normalization(
                self.tissue_rgb, W_target=self.cdt.target_W_macenko,
                mask_out=self.labeled != self.cdt.GTcodes
                    .loc["not_specified", "GT_code"],
                stain_unmixing_routine_params=self.
                cdt.stain_unmixing_routine_params)
        else:
            self.cdt._print2("%s: -- No normalization!" % self.monitorPrefix)

    def find_potentially_cellular_regions(self):
        """Find regions that are potentially cellular."""
        from scipy import ndimage
        from skimage.filters import gaussian

        mask_out = self.labeled != self.cdt.GTcodes.loc[
            "not_specified", "GT_code"]

        # deconvolvve to ge hematoxylin channel (cellular areas)
        # hematoxylin channel return shows MINIMA so we invert
        self.tissue_htx, _, _ = color_deconvolution_routine(
            self.tissue_rgb, mask_out=mask_out,
            **self.cdt.stain_unmixing_routine_params)
        self.tissue_htx = 255 - self.tissue_htx[..., 0]

        # get cellular regions by threshold HTX stain channel
        self.maybe_cellular, _ = get_tissue_mask(
            self.tissue_htx.copy(), deconvolve_first=False,
            n_thresholding_steps=1, sigma=self.cdt.cellular_step1_sigma,
            min_size=self.cdt.cellular_step1_min_size)

        # Second, low-pass filter to dilate and smooth a bit
        self.maybe_cellular = gaussian(
            0 + (self.maybe_cellular > 0), sigma=self.cdt.cellular_step2_sigma,
            output=None, mode='nearest', preserve_range=True)

        # find connected components
        self.maybe_cellular, _ = ndimage.label(self.maybe_cellular)

        # restrict cellular regions to not-otherwise-specified
        self.maybe_cellular[mask_out] = 0

        # assign to mask
        self.labeled[self.maybe_cellular > 0] = self.cdt.GTcodes.loc[
            'maybe_cellular', 'GT_code']

    def find_top_cellular_regions(self):
        """Keep largest and most cellular regions."""
        # keep only largest n regions regions
        top_cellular_mask = _get_largest_regions(
            self.maybe_cellular, top_n=self.cdt.cellular_largest_n)
        top_cellular = self.maybe_cellular.copy()
        top_cellular[top_cellular_mask == 0] = 0

        # get intensity features of hematoxylin channel for each region
        intensity_feats = compute_intensity_features(
            im_label=top_cellular, im_intensity=self.tissue_htx,
            feature_list=['Intensity.Mean'])
        unique = np.unique(top_cellular[top_cellular > 0])
        intensity_feats.index = unique

        # get top n brightest regions from the largest areas
        intensity_feats.sort_values("Intensity.Mean", axis=0, inplace=True)
        discard = np.array(intensity_feats.index[:-self.cdt.cellular_top_n])
        discard = np.in1d(top_cellular, discard).reshape(top_cellular.shape)
        top_cellular[discard] = 0

        # integrate into labeled mask
        self.labeled[top_cellular > 0] = self.cdt.GTcodes.loc[
            'top_cellular', 'GT_code']

    def visualize_results(self):
        """Visualize results in DSA."""
        # get contours
        contours_df = get_contours_from_mask(
            MASK=self.labeled, GTCodes_df=self.cdt.GTcodes.copy(),
            groups_to_get=self.cdt.groups_to_visualize,
            get_roi_contour=self.cdt.get_roi_contour, roi_group='roi',
            background_group='not_specified',
            discard_nonenclosed_background=True,
            MIN_SIZE=15, MAX_SIZE=None,
            verbose=self.cdt.verbose == 3,
            monitorPrefix=self.monitorPrefix + ": -- contours")

        # get annotation docs
        annprops = {
            'F': self.cdt.slide_info['magnification'] / self.cdt.MAG,
            'X_OFFSET': self.xmin,
            'Y_OFFSET': self.ymin,
            'opacity': self.cdt.opacity,
            'lineWidth': self.cdt.lineWidth,
        }
        annotation_docs = get_annotation_documents_from_contours(
            contours_df.copy(), separate_docs_by_group=True,
            docnamePrefix=self.cdt.docnameprefix,
            annprops=annprops, verbose=self.cdt.verbose == 3,
            monitorPrefix=self.monitorPrefix + ": -- annotation docs")

        # post annotations to slide
        for doc in annotation_docs:
            _ = self.cdt.gc.post(
                "/annotation?itemId=" + self.cdt.slide_id, json=doc)


class Cellularity_detector_thresholding(Base_HTK_Class):
    """Detect cellular regions in a slide using thresholding.

    This uses a thresholding and stain unmixing based pipeline
    to detect highly-cellular regions in a slide. The run()
    method of the CDT_single_tissue_piece() class has the key
    steps of the pipeline. In summary, here are the steps
    involved...

    1. Detect tissue from background using the RGB slide
    thumbnail. Each "tissue piece" is analysed independently
    from here onwards. The tissue_detection modeule is used
    for this step. A high sensitivity, low specificity setting
    is used here.

    2. Fetch the RGB image of tissue at target magnification. A
    low magnification (default is 3.0) is used and is sufficient.

    3. The image is converted to HSI and LAB spaces. Thresholding
    is performed to detect various non-salient components that
    often throw-off the color normalization and deconvolution
    algorithms. Thresholding includes both minimum and maximum
    values. The user can set whichever thresholds of components
    they would like. The development of this workflow was focused
    on breast cancer so the thresholded components by default
    are whote space (or adipose tissue), dark blue/green blotches
    (sharpie, inking at margin, etc), and blood. Whitespace
    is obtained by thresholding the saturation and intensity,
    while other components are obtained by thresholding LAB.

    4. Now that we know where "actual" tissue is, we do a MASKED
    color normalization to a prespecified standard. The masking
    ensures the normalization routine is not thrown off by non-
    tissue components.

    5. Perform masked stain unmixing/deconvolution to obtain the
    hematoxylin stain channel.

    6. Smooth and threshold the hematoxylin channel. Then
    perform connected component analysis to find contiguous
    potentially-cellular regions.

    7. Keep the n largest potentially-cellular regions. Then
    from those large regions, keep the m brightest regions
    (using hematoxylin channel brightness) as the final
    salient/cellular regions.

    """

    def __init__(self, gc, slide_id, GTcodes, **kwargs):
        """Init Cellularity_Detector_Superpixels object.

        Arguments:
        -----------
        gc : object
            girder client object

        slide_id : str
            girder ID of slide

        GTcodes : pandas Dataframe
            the ground truth codes and information dataframe.
            WARNING: Modified inside this method so pass a copy.
            This is a dataframe that is indexed by the annotation group name
            and has the following columns...

            group: str
                group name of annotation, eg. mostly_tumor
            overlay_order: int
                how early to place the annotation in the
                mask. Larger values means this annotation group is overlayed
                last and overwrites whatever overlaps it.
            GT_code: int
                desired ground truth code (in the mask).
                Pixels of this value belong to corresponding group (class)
            is_roi: bool
                whether this group encodes an ROI
            is_background_class: bool
                whether this group is the default fill value inside the ROI.
                For example, you may decide that any pixel inside the ROI
                is considered stroma.
            color: str
                rgb format. eg. rgb(255,0,0)

            The following indexes must be present...
            outside_tissue, not_specified, maybe_cellular, top_cellular

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

        MAG : float
            magnification at which to detect cellularity

        color_normalization_method : str
            Must be in ['reinhard', 'macenko_pca', 'none']

        target_W_macenko : np array
            3 by 3 stain matrix for macenko normalization
            obtained using rgb_separate_stains_macenko_pca()
            and reordered such that hematoxylin and eosin are
            the first and second channels, respectively.

        target_stats_reinhard : dict
            must contains the keys mu and sigma. Mean and sigma
            of target image in LAB space for reinhard normalization.

        get_tissue_mask_kwargs : dict
            kwargs for the get_tissue_mask() method. This is used
            to detect tissue from the slide thumbnail.

        keep_components : list
            list of strings. Names of components to exclude by
            HSI thresholding. These much be present in the index
            of the GTcodes dataframe

        get_tissue_mask_kwargs2 : dict
            kwargs for get_tissue_mask() used for iterative smoothing
            and thresholding the component masks after initial
            thresholding using the user-defined HSI/LAB thresholds.

        hsi_thresholds : dict
            each entry is a dict containing the keys hue, saturation
            and intensity. Each of these is in turn also a dict
            containing the keys min and max. See default value below
            for an example.

        lab_thresholds : dict
            each entry is a dict containing the keys l, a, and b.
            Each of these is in turn also a dict containing the keys
            min and max. See default value below for an example.

        stain_unmixing_routine_params : dict
            kwargs passed as the stain_unmixing_routine_params
            argument to the deconvolution_based_normalization method

        cellular_step1_sigma : float
            sigma of gaussian smoothing for first cellularity step

        cellular_step1_min_size : int
            minimum contiguous size for first cellularity step

        cellular_step2_sigma : float
            sigma of gaussian smoothing for second cellularity step

        cellular_largest_n : int
            Number of large continugous cellular regions to keep

        cellular_top_n : int
            Number of final "top" cellular regions to keep

        visualize : bool
            whether to visualize results in DSA

        opacity : float
            opacity of superpixel polygons when posted to DSA.
            0 (no opacity) is more efficient to render.

        lineWidth : float
            width of line when displaying region boundaries.

        docnameprefix : str
            prefix to add to annotation document name

        groups_to_visualize : list
            which groups to visualize

        get_roi_contour : bool
            whether to get the contour of the roi

        """
        default_attr = {

            # The following are already assigned defaults by Base_HTK_Class
            # 'verbose': 1,
            # 'monitorPrefix': "",
            # 'logging_savepath': None,
            # 'suppress_warnings': False,

            'MAG': 3.0,

            # Must be in ['reinhard', 'macenko_pca', 'none']
            'color_normalization_method': 'macenko_pca',

            # TCGA-A2-A3XS-DX1_xmin21421_ymin37486_.png, Amgad et al, 2019)
            # is used as the target image for reinhard & macenko normalization

            # for macenco (obtained using rgb_separate_stains_macenko_pca()
            # and using reordered such that columns are the order:
            # Hamtoxylin, Eosin, Null
            'target_W_macenko': np.array([
                [0.5807549,  0.08314027,  0.08213795],
                [0.71681094,  0.90081588,  0.41999816],
                [0.38588316,  0.42616716, -0.90380025]
            ]),

            # TCGA-A2-A3XS-DX1_xmin21421_ymin37486_.png, Amgad et al, 2019)
            # Reinhard color norm. standard
            'target_stats_reinhard': {
                'mu': np.array([8.74108109, -0.12440419,  0.0444982]),
                'sigma': np.array([0.6135447, 0.10989545, 0.0286032]),
            },

            # kwargs for getting masks for all tissue pieces (thumbnail)
            'get_tissue_mask_kwargs': {
                'deconvolve_first': True, 'n_thresholding_steps': 1,
                'sigma': 1.5, 'min_size': 500,
            },

            # components to extract by HSI thresholding
            'keep_components': ['blue_sharpie', 'blood', 'whitespace', ],

            # kwargs for getting components masks
            'get_tissue_mask_kwargs2': {
                'deconvolve_first': False, 'n_thresholding_steps': 1,
                'sigma': 5.0, 'min_size': 50,
            },

            # min/max thresholds for HSI and LAB
            'hsi_thresholds': {
                'whitespace': {
                    'hue': {'min': 0, 'max': 1.0},
                    'saturation': {'min': 0, 'max': 0.2},
                    'intensity': {'min': 220, 'max': 255},
                },
            },
            'lab_thresholds': {
                'blue_sharpie': {
                    'l': {'min': -1000, 'max': 1000},
                    'a': {'min': -1000, 'max': 1000},
                    'b': {'min': -1000, 'max': -0.02},
                },
                'blood': {
                    'l': {'min': -1000, 'max': 1000},
                    'a': {'min': 0.02, 'max': 1000},
                    'b': {'min': -1000, 'max': 1000},
                },
            },

            # for stain unmixing to deconvolove and/or color normalize
            'stain_unmixing_routine_params': {
                'stains': ['hematoxylin', 'eosin'],
                'stain_unmixing_method': 'macenko_pca',
            },

            # params for getting cellular regions
            'cellular_step1_sigma': 0.,
            'cellular_step1_min_size': 100,
            'cellular_step2_sigma': 1.5,
            'cellular_largest_n': 5,
            'cellular_top_n': 2,

            # visualization params
            'visualize': True,
            'opacity': 0,
            'lineWidth': 3.0,
            'docnameprefix': 'cdt',
            'groups_to_visualize': None,  # everything
            'get_roi_contour': True,
        }
        default_attr.update(kwargs)
        super().__init__(
            default_attr=default_attr)

        self.color_normalization_method = \
            self.color_normalization_method.lower()
        assert self.color_normalization_method in [
            'reinhard', 'macenko_pca', 'none']

        # set attribs
        self.gc = gc
        self.slide_id = slide_id
        self.GTcodes = GTcodes

        self.fix_GTcodes()

    def fix_GTcodes(self):
        """Fix self.GTcodes (important!)."""
        # validate
        self.GTcodes.index = self.GTcodes.loc[:, "group"]
        necessary_indexes = self.keep_components + [
            'outside_tissue', 'not_specified',
            'maybe_cellular', 'top_cellular']
        assert all(j in list(self.GTcodes.index) for j in necessary_indexes)

        # Make sure the first things layed out are the "background" components
        min_val = np.min(self.GTcodes.loc[:, 'overlay_order'])
        self.GTcodes.loc['outside_tissue', 'overlay_order'] = min_val - 2
        self.GTcodes.loc['not_specified', 'overlay_order'] = min_val - 1

        # reorder in overlay order (important)
        self.GTcodes.sort_values('overlay_order', axis=0, inplace=True)
        self.ordered_components = list(self.GTcodes.loc[:, "group"])

        # only keep relevant components (for HSI/LAB thresholding)
        for c in self.ordered_components[:]:
            if c not in self.keep_components:
                self.ordered_components.remove(c)

    def run(self):
        """Run full pipeline to detect cellular regions."""
        # get mask, each unique value is a single tissue piece
        self._print1(
            "%s: set_slide_info_and_get_tissue_mask()" % self.monitorPrefix)
        labeled = self.set_slide_info_and_get_tissue_mask()

        # Go through tissue pieces and do run sequence
        unique_tvals = list(set(np.unique(labeled)) - {0, })
        tissue_pieces = [None for _ in range(len(unique_tvals))]
        for idx, tval in enumerate(unique_tvals):
            monitorPrefix = "%s: Tissue piece %d of %d" % (
                self.monitorPrefix, idx+1, len(unique_tvals))
            self._print1(monitorPrefix)
            tissue_pieces[idx] = CDT_single_tissue_piece(
                self, tissue_mask=labeled == tval, monitorPrefix=monitorPrefix)
            tissue_pieces[idx].run()
            # delete unnecessary attributes
            del (
                tissue_pieces[idx].tissue_rgb,  # too much space
                tissue_pieces[idx].tissue_mask,  # already part of labeled
                tissue_pieces[idx].maybe_cellular,  # already part of labeled
                tissue_pieces[idx].tissue_htx,  # unnecessary
            )

        return tissue_pieces

    def set_color_normalization_target(
            self, ref_image_path, color_normalization_method='macenko_pca'):
        """Set color normalization values to use from target image.

        Arguments
        ref_image_path, str
        >    path to target (reference) image
        color_normalization_method, str
        >    color normalization method to use. Currently, only
        >    reinhard and macenko_pca are accepted.

        """
        from imageio import imread

        # read input image
        ref_im = np.array(imread(ref_image_path, pilmode='RGB'))

        # assign target values

        color_normalization_method = color_normalization_method.lower()

        if color_normalization_method == 'reinhard':
            mu, sigma = lab_mean_std(ref_im)
            self.target_stats_reinhard['mu'] = mu
            self.target_stats_reinhard['sigma'] = sigma

        elif color_normalization_method == 'macenko_pca':
            self.target_W_macenko = _reorder_stains(
                rgb_separate_stains_macenko_pca(ref_im, I_0=None),
                stains=['hematoxylin', 'eosin'])
        else:
            raise ValueError(
                "Unknown color_normalization_method: %s" %
                (color_normalization_method))

        self.color_normalization_method = color_normalization_method

    def set_slide_info_and_get_tissue_mask(self):
        """Set self.slide_info dict and self.labeled tissue mask."""
        # This is a presistent dict to store information about slide
        self.slide_info = self.gc.get('item/%s/tiles' % self.slide_id)

        # get tissue mask
        thumbnail_rgb = get_slide_thumbnail(self.gc, self.slide_id)

        # get labeled tissue mask -- each unique value is one tissue piece
        labeled, _ = get_tissue_mask(
            thumbnail_rgb, **self.get_tissue_mask_kwargs)

        if len(np.unique(labeled)) < 2:
            raise ValueError("No tissue detected!")

        # Find size relative to WSI
        self.slide_info['F_tissue'] = self.slide_info[
            'sizeX'] / labeled.shape[1]

        return labeled
