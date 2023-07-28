from concurrent.futures import ProcessPoolExecutor

import shapely
from shapely.geometry import Polygon


def create_polygon(coordinates):
    """
    Create a shapely Polygon from a list of points.

    Args:
        points (list): A list containing tuples representing the points of the polygon.

    Returns:
        shapely.geometry.Polygon: The polygon created from the points.
    """
    return Polygon(coordinates).buffer(0)


def convert_polygons_tobbox(nuclei_list):
    """
    Convert nuclei segmentation data from polygon format to bounding box format.

    Parameters:
    nuclei_list (list): A list of dictionaries, where each dictionary represents a nuclei object.

    Returns:
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


def remove_overlap_nuclei(nuclei_list, nuclei_format):
    """
    Remove overlapping nuclei from the given list using parallel processing.

    This function creates a single shapely STRtree from all the polygons in the nuclei_list
    and filters out the overlapping polygons.

    Args:
        nuclei_list (list): A list of dictionaries, each containing 'points' representing a polygon.

    Returns:
        output_list (list): A new list with overlapping nuclei removed.
    """

    # Use ProcessPoolExecutor for parallel processing of polygon creation
    with ProcessPoolExecutor() as executor:
        polygons = list(
            executor.map(
                create_polygon,
                (nuclei['points'] for nuclei in nuclei_list)))

    # Build the STRtree from all the polygons
    rt = shapely.strtree.STRtree(polygons)

    # Find and remove any overlapping polygons
    output_list = [nuclei for index, nuclei in enumerate(nuclei_list) if not any(
        ix for ix in rt.query(
            polygons[index]) if ix < index and polygons[index].intersects(
            polygons[ix]))]

    # if nuclei_format is bbox - convert polygons to bbox
    if nuclei_format == 'bbox':
        output_list = convert_polygons_tobbox(output_list)

    return output_list
