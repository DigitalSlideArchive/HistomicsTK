import os
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
        list: A new list with overlapping nuclei removed.
    """
    # Extract all points from the nuclei_list to create polygons
    coordinates_collection = [nuclei['points'] for nuclei in nuclei_list]

    # Use ProcessPoolExecutor for parallel processing of polygon creation
    with ProcessPoolExecutor(os.cpu_count()) as executor:
        polygons = list(
            executor.map(
                create_polygon,
                (coordinates for coordinates in coordinates_collection)))

    # Build the STRtree from all the polygons
    rt = shapely.strtree.STRtree(polygons)

    # Find indexes of overlapping polygons
    remove_list = [
        index for index in range(
            len(polygons)) if any(
            ix for ix in rt.query(
                polygons[index]) if ix < index and polygons[index].intersects(
                    polygons[ix]))]

    # Remove the overlapping polygons from the original list
    output_list = [nuclei_list[i] for i in range(len(nuclei_list)) if i not in remove_list]

    return output_list
