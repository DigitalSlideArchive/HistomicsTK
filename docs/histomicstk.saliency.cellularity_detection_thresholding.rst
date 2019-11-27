Cellularity Detection using thresholding
=======================================================

**Overview:** 

This uses a thresholding and stain unmixing based pipeline
to detect highly-cellular regions in a slide. The run()
method of the CDT_single_tissue_piece() class has the key
steps of the pipeline. 

**Implementation summary:** 

1. Detect tissue from background using the RGB slide
thumbnail. Each "tissue piece" is analysed independently
from here onwards. The tissue_detection module is used
for this step. A high sensitivity, low specificity setting
is used here.

2. Fetch the RGB image of tissue at target magnification. A
low magnification (default is 3.0) is used and is sufficient.

3. The image is converted to HSI and LAB spaces. Thresholding
is performed to detect various non-salient components that
often throw-off the color normalization and deconvolution
algorithms. Thresholding includes both minimum and maximum
values. The user can set whichever thresholds of components
they would like. The development of this workflow was focused
on breast cancer so the thresholded components by default
are whote space (or adipose tissue), dark blue/green blotches
(sharpie, inking at margin, etc), and blood. Whitespace
is obtained by thresholding the saturation and intensity,
while other components are obtained by thresholding LAB.

4. Now that we know where "actual" tissue is, we do a MASKED
color normalization to a prespecified standard. The masking
ensures the normalization routine is not thrown off by non-
tissue components.

5. Perform masked stain unmixing/deconvolution to obtain the
hematoxylin stain channel.

6. Smooth and threshold the hematoxylin channel. Then
perform connected component analysis to find contiguous
potentially-cellular regions.

7. Keep the n largest potentially-cellular regions. Then
from those large regions, keep the m brightest regions
(using hematoxylin channel brightness) as the final
salient/cellular regions.

These slides are used as a test examples:

`TCGA-A2-A0YE-01Z-00-DX1 <http://candygram.neurology.emory.edu:8080/histomicstk#?image=5d586d76bd4404c6b1f286ae>`_
and
`TCGA-A1-A0SK-01Z-00-DX1 <http://candygram.neurology.emory.edu:8080/histomicstk#?image=5d817f5abd4404c6b1f744bb>`_

**Here's the result:**

.. image:: https://user-images.githubusercontent.com/22067552/67629132-3dd1bc80-f848-11e9-8217-856ecf5b2801.png
   :target: https://user-images.githubusercontent.com/22067552/67629132-3dd1bc80-f848-11e9-8217-856ecf5b2801.png
   :alt: cdetection


**Where to look?**

::

   |_ histomicstk/
   |   |_saliency/
   |      |_cellularity_detection_thresholding.py 
   |      |_tests/
   |         |_cellularity_detection_thresholding_test.py
   |_ docs/
       |_examples/
          |_cellularity_detection_thresholding.ipynb

.. automodule:: histomicstk.saliency.cellularity_detection_thresholding
    :members:
    :undoc-members:
    :show-inheritance:
