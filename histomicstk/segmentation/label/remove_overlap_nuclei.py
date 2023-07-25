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


def remove_overlap_nuclei(nuclei_list):
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

    return output_list
