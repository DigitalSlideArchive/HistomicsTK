Polygon merger (overview)
===========================

There are two versions, depending on the user's requirements: ``polygon_merger`` and ``polygon_merger_v2``. ``polygon_merger`` deals with the situation when you have tiled masks and want to fuse the contours from tiled edges. ``polygon_merger_v2`` is a more general version that does not need masks. In brief, here is when it is appropriate to use either workflow. If you have:


* 
  **Tiled masks from segmentation algorithm**\ : Use ``polygon_merger.py``. That is likely going to be **faster** than ``polygon_merger_v2.py``\ , since it takes advantage of the known spatial adjacency between segmentation masks and the available functionality in ``masts_to_annotations_handler.py`` to cut down the search space to a much smaller set of annotations (only those that touch the mak edge). ``polygon_merger_test.py`` runs in ~20 seconds whereas ``polygon_merger_v2_test.py`` runs in ~25 seconds (i.e. 20% slower, but it's not a 100% fair comparison). 

* 
  **Contours (coordinates) from whole slide image**\ : Use ``polygon_merger_v2.py`` (as shown in this notebook). This is **more general** and does not require having masks. It is still quite fast since it relies on R-trees to cut down the search space.

Details on each method and use case are provided below.

This extends on some of the workflows described in Amgad et al, 2019:

*Mohamed Amgad, Habiba Elfandy, Hagar Hussein, ..., Jonathan Beezley, Deepak R Chittajallu, David Manthey, David A Gutman, Lee A D Cooper, Structured crowdsourcing enables convolutional segmentation of histology images, Bioinformatics, 2019, btz083*

This slide used as a test example:

`TCGA-A2-A0YE-01Z-00-DX1 <http://candygram.neurology.emory.edu:8080/histomicstk#?image=5d5d6910bd4404c6b1f3d893&bounds=41996%2C43277%2C49947%2C46942%2C0>`_

**This is what the result looks like:**


.. image:: https://user-images.githubusercontent.com/22067552/63630235-866dbf00-c5e6-11e9-94d4-02d736e06f15.png
   :target: https://user-images.githubusercontent.com/22067552/63630235-866dbf00-c5e6-11e9-94d4-02d736e06f15.png
   :alt: image

Polygon merger (version 1 - from tiled masks)
===============================================

**Overview:**

This includes functionality to merge polygons from adjacent tiles (regions of interest, ROI) that form a tiled array. This is particularly useful when working with algorithmic segmentation output, which typically produces segmentation masks that are much smaller in size (say, 512x512 pixels) relative to the tissue area. This means that at the edge of each tile the segmentation contour discontinues sharply. This is very problematic if you need to analyze histological structures that are very large, oftentimes thousands of pixels perimeter-wise. 

**Implementation summary**

The key requirement is that that the masks (ROIs) are rectangular and unrotated, everything else is taken care of. This algorithm fuses polygon clusters in coordinate (not mask) space, which means it can merge almost-arbitrarily large structures without memory issues. The algorithm, in brief, works as follows:


* 
  Extract contours from the given masks using functionality from the ``masks_to_annotations_handler.py``\ , making sure to correctly account for contour offset, so that all coordinates are relative to whole-slide image.

* 
  Identify contours that touch the edge of each ROI and which edges they touch. 

* 
  Identify shared edges between ROIs (using 4-connectivity). 

* 
  For each shared edge, find contours that are within the vicinity of each other (using bounding box location). If they are, then convert to shapely polygon and check if they actually are within a threshold distance of each other. If so, consider them a "pair" for merger.

* 
  Hierarchically cluster pairs of polygons such that all contiguous polygons are to be merged.

* 
  Get the union of each polygon "cluster" elements. The polygons are first dilated a bit to make sure any small gaps are covered, then they are merged and eroded.

This initial set of "vetting" steps ensures that the complexity is ``<< O(n^2)``. This is very important since time complexity plays a key role as whole slide images may contain tens of thousands of objects.

**Where to look?**

::

   |_ histomicstk/
   |   |_annotations_and_masks/
   |      |_polygon_merger.py 
   |      |_tests/
   |         |_ polygon_merger_test.py
   |         |_test_files/
   |            |_polygon_merger_roi_masks/
   |_ docs/
       |_examples/
          |_polygon_merger.ipynb

----

.. automodule:: histomicstk.annotations_and_masks.polygon_merger
    :members:
    :undoc-members:
    :show-inheritance:

Polygon merger (version 2- general)
=====================================

**Overview:**

This includes functionality to merge polygons from a whole-slide image using an R-tree spatial querying database. This module is particularly useful when working with algorithmic segmentation outputs that are small relative to the tissue area. If you need to analyze histological structures that are very large, oftentimes thousands of pixels perimeter-wise, this will be a useful functionality. 

**Implementation summary**

This algorithm fuses polygon clusters in coordinate (not mask) space, which means is can merge almost-arbitrarily large structures without memory issues. The algorithm, in brief, works as follows:


* 
  Identify contours that that have the same label (eg. tumor)

* 
  Add the bounding boxes from these contours to an `R-tree <https://en.wikipedia.org/wiki/R-tree>`_. The R-tree implementation used here is modified from `here <https://code.google.com/archive/p/pyrtree/>`_\ , and uses k-means clustering to balance the tree. 

* 
  Starting from the bottom of the tree, merge all polygons from leafs that belong to the same nodes.

* 
  Move one level up the hierarchy, each time incorporated merged polygons from nodes that share a common parent. Do this until you get one merged polygon at the root node. The polygons are first dilated a bit to make sure any small gaps are covered, then they are merged and eroded. Note that ``shapely`` allows us to simply "combine" polygons into a "multi-polygon" object, which means we do not need to check if the polygons actually are within merger threshold. Instead, all polygons hierarchically combined until we arrive at a single "multi-polygon" object at the root node.

* 
  Save the coordinates from each polygon on the multi-polygon object in a new pandas DataFrame. 

This process ensures that the number of comparisons is ``<< n^2``. This is very important since algorithm complexity plays a key role as whole slide images may contain tens of thousands of objects.

**Where to look?**

::

   |_ histomicstk/
   |   |_annotations_and_masks/
   |      |_polygon_merger_v2.py 
   |      |_tests/
   |         |_ polygon_merger_v2_test.py
   |_ docs/
       |_examples/
          |_polygon_merger_v2.ipynb

.. automodule:: histomicstk.annotations_and_masks.polygon_merger_v2
    :members:
    :undoc-members:
    :show-inheritance:
