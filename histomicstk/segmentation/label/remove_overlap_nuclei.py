import shapely


def create_polygon(coordinates):
    """
    Create a shapely Polygon from a list of points.

    Args:
    ----
        points (list): A list containing tuples representing the points of the polygon.

    Returns:
    -------
        shapely.geometry.Polygon: The polygon created from the points.

    """
    return shapely.geometry.Polygon(coordinates).buffer(0)


def convert_polygons_tobbox(nuclei_list):
    """
    Convert nuclei segmentation data from polygon format to bounding box format.

    Parameters
    ----------
    nuclei_list (list): A list of dictionaries, where each dictionary represents a nuclei object.

    Returns
    -------
    list: The modified 'nuclei_list', where each nuclei object has been transformed
    to a bounding box representation.

    """
    for nuclei in nuclei_list:
        # Set 'type' property of the nuclei to 'rectangle'
        nuclei['type'] = 'rectangle'
        nuclei['rotation'] = 0

        # Calculate the minimum and maximum x and y coordinates to obtain the bounding box.
        minx = min(p[0] for p in nuclei['points'])
        maxx = max(p[0] for p in nuclei['points'])
        miny = min(p[1] for p in nuclei['points'])
        maxy = max(p[1] for p in nuclei['points'])

        # Calculate the center of the bounding box, width, height
        nuclei['center'] = [(minx + maxx) / 2, (miny + maxy) / 2, 0]
        nuclei['width'] = maxx - minx
        nuclei['height'] = maxy - miny

        # Remove the 'closed' and 'points'
        nuclei.pop('closed')
        nuclei.pop('points')

    return nuclei_list


def remove_overlap_nuclei(nuclei_list, nuclei_format, return_selected_nuclei=False):
    """
    Remove overlapping nuclei using spatial indexing and parallel processing.

    This function removes overlapping nuclei polygons from input using an STRtree index.

    Args:
    ----
        nuclei_list (list): List of dictionaries with 'points' for polygons.
        nuclei_format (str, optional): Output format ('polygon' or 'bbox').
        return_selected (bool, optional): Return indices of selected nuclei (default: False).

    Returns:
    -------
        output_list (list): New list with overlapping nuclei removed.
        selected_nuclei (list, optional): Indices of selected nuclei.

    """
    polygons = [create_polygon(nuclei['points']) for nuclei in nuclei_list]

    # Build the STRtree from all the polygons
    rt = shapely.strtree.STRtree(polygons)

    # Find and remove any overlapping polygons
    output = [(nuclei, index) for index, nuclei in enumerate(nuclei_list) if not any(
        ix for ix in rt.query(polygons[index])
        if ix < index and polygons[index].intersects(polygons[ix]))]

    output_list, selected_nuclei = (
        zip(*output) if return_selected_nuclei else ([nuclei for nuclei, _ in output], []))

    # if nuclei_format is bbox - convert polygons to bbox
    if nuclei_format == 'bbox':
        output_list = convert_polygons_tobbox(output_list)

    if return_selected_nuclei:
        return output_list, selected_nuclei

    return output_list
