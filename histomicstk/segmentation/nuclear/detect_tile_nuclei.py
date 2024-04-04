import numpy as np

import histomicstk.features as htk_features
import histomicstk.preprocessing.color_deconvolution as htk_cdeconv
import histomicstk.preprocessing.color_normalization as htk_cnorm
import histomicstk.segmentation.label as htk_seg_label
import histomicstk.segmentation.nuclear as htk_nuclear
from histomicstk.cli import utils as cli_utils


def detect_tile_nuclei(tile_info, args, src_mu_lab=None,
                       src_sigma_lab=None, invert_image=False,
                       default_img_inversion=False, return_fdata=False):
    """
    Detect nuclei within a tile image and generate annotations.

    Args:
    ----
        tile_info (dict): Information about the tile image.
        args: Arguments from the cli.
        src_mu_lab: Source mean in LAB color space.
        src_sigma_lab: Source standard deviation in LAB color space.
        invert_image (bool): Invert the image.
        default_img_inversion (bool): Apply default image inversion.
        return_fdata (bool): Return computed features.

    Returns:
    -------
        nuclei_annot_list (list): List of nuclei annotations.
        fdata: Computed nuclei features.

    """
    # Flags
    single_channel = False

    # get tile image & check number of channels
    single_channel = len(tile_info['tile'].shape) <= 2 or tile_info['tile'].shape[2] == 1
    if single_channel:
        im_tile = np.dstack((tile_info['tile'], tile_info['tile'], tile_info['tile']))
        if default_img_inversion:
            invert_image = True
    else:
        im_tile = tile_info['tile'][:, :, :3]

    # perform image inversion
    if invert_image:
        im_tile = (np.iinfo(im_tile.dtype).max
                   if im_tile.dtype.kind == 'u'
                   else np.max(im_tile)) - im_tile

    # perform color normalization
    im_nmzd = htk_cnorm.reinhard(im_tile,
                                 args.reference_mu_lab,
                                 args.reference_std_lab,
                                 src_mu=src_mu_lab,
                                 src_sigma=src_sigma_lab)

    # perform color decovolution
    w = cli_utils.get_stain_matrix(args)

    # perform deconvolution
    im_stains = htk_cdeconv.color_deconvolution(im_nmzd, w).Stains
    im_nuclei_stain = im_stains[:, :, 0].astype(float)

    # segment nuclear foreground
    im_nuclei_fgnd_mask = im_nuclei_stain < args.foreground_threshold

    # segment nuclei
    im_nuclei_seg_mask = htk_nuclear.detect_nuclei_kofahi(
        im_nuclei_stain,
        im_nuclei_fgnd_mask,
        args.min_radius,
        args.max_radius,
        args.min_nucleus_area,
        args.local_max_search_radius,
    )

    # Delete border nuclei
    if args.ignore_border_nuclei is True:

        im_nuclei_seg_mask = htk_seg_label.delete_border(im_nuclei_seg_mask)

    # Delete overlapping border nuclei
    if any(tile_info['tile_overlap'].values()) > 0:

        im_nuclei_seg_mask = htk_seg_label.delete_overlap(
            im_nuclei_seg_mask, overlap_info=tile_info['tile_overlap'])

    # generate nuclei annotations
    nuclei_annot_list = []

    flag_nuclei_found = np.any(im_nuclei_seg_mask)

    # compute nuclei features and nuclei_annot_list
    fdata = None

    if flag_nuclei_found:
        if args.nuclei_annotation_format:
            format = args.nuclei_annotation_format
            if args.nuclei_annotation_format == 'bbox' and - \
                    args.remove_overlapping_nuclei_segmentation:
                format = 'boundary'

        if return_fdata:
            if args.cytoplasm_features:
                im_cytoplasm_stain = im_stains[:, :, 1].astype(float)
            else:
                im_cytoplasm_stain = None
            # Generate features and nuclei annotation simultaneously
            fdata, nuclei_annot_list = htk_features.compute_nuclei_features(
                im_nuclei_seg_mask,
                im_nuclei_stain.astype(np.uint8), im_cytoplasm_stain.astype(np.uint8),
                fsd_bnd_pts=args.fsd_bnd_pts,
                fsd_freq_bins=args.fsd_freq_bins,
                cyto_width=args.cyto_width,
                num_glcm_levels=args.num_glcm_levels,
                morphometry_features_flag=args.morphometry_features,
                fsd_features_flag=args.fsd_features,
                intensity_features_flag=args.intensity_features,
                gradient_features_flag=args.gradient_features,
                tile_info=tile_info,
                im_nuclei_seg_mask=im_nuclei_seg_mask,
                format=format,
                return_nuclei_annotation=True,
            )
            fdata.columns = ['Feature.' + col for col in fdata.columns]
        else:
            nuclei_annot_list = cli_utils.create_tile_nuclei_annotations(
                im_nuclei_seg_mask, tile_info, format)

    if return_fdata:
        return nuclei_annot_list, fdata

    return nuclei_annot_list
