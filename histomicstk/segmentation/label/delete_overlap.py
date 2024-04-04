import numpy as np

from .condense import condense
from .delete import delete


def delete_overlap(im_label, overlap_info):
    """
    Deletes overlapping regions from an image label based on overlap information and tile size.

    Args:
    ----
        im_label (ndarray): Image label represented as a NumPy array.
        overlap_info (dict): Dictionary containing overlap information.
                            It should have the following keys:
                            'left', 'right', 'top', and 'bottom',
                            each specifying the overlap amount in pixels.

    Returns:
    -------
        ndarray: Image label with overlapping regions deleted.

    Note:
    ----
        This function assumes the necessary imports, such as `np`, are already present.

    """
    # Compute the half of the overlap values
    left_overlap = overlap_info['left'] // 2
    right_overlap = overlap_info['right'] // 2
    top_overlap = overlap_info['top'] // 2
    bottom_overlap = overlap_info['bottom'] // 2

    if not np.any(im_label):
        return im_label

    # Create a border mask based on the overlap values
    im_border_mask = np.zeros_like(im_label)
    if left_overlap:
        im_border_mask[:, :left_overlap] = True
    if right_overlap:
        im_border_mask[:, -right_overlap:] = True
    if top_overlap:
        im_border_mask[:top_overlap, :] = True
    if bottom_overlap:
        im_border_mask[-bottom_overlap:, :] = True

    # Find unique indices of the border regions
    border_indices = np.unique(im_label[im_border_mask > 0])
    border_indices = border_indices[border_indices > 0]

    if len(border_indices) == 0:
        return im_label

    # Condense and delete the overlap regions from the image label
    im_label_del = np.zeros_like(im_label)
    im_label_del = condense(delete(im_label, border_indices))
    return im_label_del
