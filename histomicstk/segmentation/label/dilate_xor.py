import numpy as np


def dilate_xor(im_label, neigh_width=8):
    """Computes a label mask highlighting a ring-like neighborhood of each
    object or region in a given label mask

    Parameters
    ----------
    im_label : array_like
        A labeled mask image wherein intensity of a pixel is the ID of the
        object it belongs to. Non-zero values are considered to be foreground
        objects.

    neigh_width : float, optional
        The width of the ring-like neighborhood around each object.

    Returns
    -------
    im_neigh_label : array_like
        A labeled mask image highlighting pixels in a ring-like neighborhood of
        width upto `neigh_width` around each object in the given label mask.
        The intensity of each pixel in the ring-like neighborhood is set
        equal to the label of the closest object in the given label mask.
        other pixels (including the ones inside objects) are set to zero.

    """
    from scipy.ndimage import distance_transform_edt

    # For each background pixel compute the distance to the nearest object and
    # the indices of the nearest object pixel
    im_dist, closest_obj_ind = distance_transform_edt(im_label == 0,
                                                      return_indices=True)
    closest_obj_rind, closest_obj_cind = closest_obj_ind

    # Get indices of background pixels within a given distance from an object
    neigh_rind, neigh_cind = np.where(
        np.logical_and(im_dist > 0, im_dist <= neigh_width),
    )

    # generate labeled neighborhood mask
    im_neigh_label = np.zeros_like(im_label)

    im_neigh_label[neigh_rind, neigh_cind] = im_label[
        closest_obj_rind[neigh_rind, neigh_cind],
        closest_obj_cind[neigh_rind, neigh_cind]]

    return im_neigh_label
