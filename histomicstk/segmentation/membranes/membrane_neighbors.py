import numpy as np
import scipy as sp

import skimage.morphology
import skimage.measure
import skimage.draw

from collections import defaultdict


def membrane_neighbors(m_label, h_label, branches, length=25,
                       theta=1, b_dilation=3):
    """Finds the nearest membrane neighbors for each nuclei.
    Takes as input a nuclei labeled image, a membrane labeled image, and
    the membrane branches and performs the detection of membranes for
    each nuclei using line segmentation with degrees.

    Parameters
    ----------
    m_label : array_like
        A membrane-labeled image.
    h_label : array_like
        A nuclei-labeled image.
    branches : array_like
        A membrane branch mask.
    length : int
        Scaling property of line segmentation. Start points of the length is
        the center of nuclei. Default value = 25.
    theta : int
        Scaling property of line segmentation. Degree between line segments.
        Default value = 1.
    b_dilation : int
        Branch dilation factor. Default value = 3.

    Returns
    -------
    m_dict : dictionary
        A collection of the nuclei and membrane labels.
        Format: [(nuclei label, list of membrane labels)]
    """

    # get membrane mask
    m_mask = np.zeros_like(m_label)
    m_mask[m_label > 0] = 1

    # get branch-dilated mask
    branches = sp.ndimage.binary_dilation(
        branches,
        structure=skimage.morphology.disk(b_dilation)
    )

    m_mask = m_mask | branches

    # set a dictionary
    m_dict = defaultdict(list)

    if theta > 0:
        # set degrees between line segments
        degrees = np.arange(360/theta)*theta

        # perform regionprop for each nuclei
        rprops = skimage.measure.regionprops(h_label)
        numLabels = len(rprops)

        for i in range(numLabels):

            # find centroids for nuclei
            cx = rprops[i].centroid[0].astype(int)
            cy = rprops[i].centroid[1].astype(int)

            # remove cx, cy crossing over membrane_mask
            if m_mask[cx, cy] is False:

                # find end points
                endx = cx + np.round(length * np.cos(np.radians(degrees)))
                endy = cy + np.round(length * np.sin(np.radians(degrees)))

                labels = []

                for j in range(len(endx)):

                    # get line-index
                    lcols, lrows = skimage.draw.line(
                        cy, cx, endy[j].astype(int), endx[j].astype(int)
                    )

                    # remove negative line-index
                    rows = lrows[(lrows >= 0) & (lrows < m_label.shape[0])]
                    cols = lcols[(lcols >= 0) & (lcols < m_label.shape[1])]

                    lsize = min(len(rows), len(cols))

                    rows = rows[:lsize]
                    cols = cols[:lsize]

                    # find nearest membrane
                    coords = []

                    for k in range(len(rows)):
                        if m_mask[rows[k], cols[k]] > 0:
                            coords.append((rows[k], cols[k]))

                    if coords:
                        nearest_coords = coords[
                            sp.spatial.distance.cdist([(cx, cy)], coords).argmin()
                        ]
                        labels.append(m_label[nearest_coords])

                if labels:
                    labels = np.unique(labels).tolist()
                    m_dict[rprops[i].label] = labels

    return m_dict
