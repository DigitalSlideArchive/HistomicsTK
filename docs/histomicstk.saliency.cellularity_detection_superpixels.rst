Cellularity Detection using superpixel classification
=======================================================

**Overview:** 

Detect cellular regions in a slides by classifying superpixels.

This uses Simple Linear Iterative Clustering (SLIC) to get superpixels at
a low slide magnification to detect cellular regions. The first step of
this pipeline detects tissue regions (i.e. individual tissue pieces)
using the get_tissue_mask method of the histomicstk.saliency module. Then,
each tissue piece is processed separately for accuracy and disk space
efficiency. It is important to keep in mind that this does NOT rely on a
tile iterator, but loads the entire tissue region (but NOT the whole slide)
in memory and passes it on to skimage.segmentation.slic method. Not using
a tile iterator helps keep the superpixel sizes large enough to correspond
to tissue boundaries.

Once superpixels are segmented, the image is deconvolved and features are
extracted from the hematoxylin channel. Features include intensity and
possibly also texture features. Then, a mixed component Gaussian mixture
model is fit to the features, and median intensity is used to rank
superpixel clusters by 'cellularity' (since we are working with the
hematoxylin channel).

Additional functionality includes contour extraction to get the final
segmentation boundaries of cellular regions and to visualize them in DSA
using one's preferred colormap.

These slides are used as a test examples:

`TCGA-A2-A0YE-01Z-00-DX1 <http://candygram.neurology.emory.edu:8080/histomicstk#?image=5d586d76bd4404c6b1f286ae>`_
and
`TCGA-A1-A0SK-01Z-00-DX1 <http://candygram.neurology.emory.edu:8080/histomicstk#?image=5d817f5abd4404c6b1f744bb>`_

**Here's the result:**

From left to right: Slide thumbnail, superpixel classifications, contiguous cellular/acellular regions


.. image:: https://user-images.githubusercontent.com/22067552/65730355-7e92b600-e08f-11e9-918a-507f117f6d77.png
   :target: https://user-images.githubusercontent.com/22067552/65730355-7e92b600-e08f-11e9-918a-507f117f6d77.png
   :alt: cdetection


**Where to look?**

::

   |_ histomicstk/
   |   |_saliency/
   |      |_cellularity_detection_superpixels.py 
   |      |_tests/
   |         |_cellularity_detection_superpixels_test.py
   |_ docs/
       |_examples/
          |_cellularity_detection_superpixels.ipynb

.. automodule:: histomicstk.saliency.cellularity_detection_superpixels
    :members:
    :undoc-members:
    :show-inheritance:
