Annotations to masks workflow
=============================

**Overview:** 

This includes tools to parse annotations from an item (slide) into masks to use in training and evaluating imaging algorithms. Two "versions" of this workflow exist:

* Get labeled mask for any region in a whole-slide image (user-defined)

* Get labeled mask for areas enclosed within special "region-of-interest" (ROI) annotations that have been drawn by the user. This involves mapping annotations (rectangles/polygons) to ROIs and making one mask per ROI.

This extends on some of the workflows described in Amgad et al, 2019:

*Mohamed Amgad, Habiba Elfandy, Hagar Hussein, ..., Jonathan Beezley, Deepak R Chittajallu, David Manthey, David A Gutman, Lee A D Cooper, Structured crowdsourcing enables convolutional segmentation of histology images, Bioinformatics, , btz083, https://doi.org/10.1093/bioinformatics/btz083*

This slide used as a test example:

`TCGA-A2-A0YE-01Z-00-DX1 <http://candygram.neurology.emory.edu:8080/histomicstk#?image=5d586d57bd4404c6b1f28640&bounds=53566%2C33193%2C68926%2C40593%2C0>`_

The user uses a csv file like the one in 
```histomicstk/annotations_and_masks/tests/test_files/sample_GTcodes.csv ```
to control pixel values assigned to mask, overlay order of various annotation groups, which groups are considered to be ROIs, etc. Note that we use the girder definition of term "group" here, which is an annotation style indicating a certain class, such as "tumor" or "necrosis".

This adds a lot of functionality on top of API endpoints that get annotations as a list of dictionaries, including handing the following complex situations:

* Getting RGB images and labeled masks at the same magnification/resolution

* User-defined regions to get, including "cropping" of annotations to desired bounds

* Getting user-drawn ROIs, including rotated rectangles and polygons

* Overlapping annotations

* "Background" class (eg. anything not-otherwise-specified is stroma)

* Getting contours and bounding boxes relative to images at the same resolution, to be used to trainign object localization models like Faster-RCNN. 


**There are four run modes:**

* **wsi**: get scaled up/down version of mask of whole slide

* **min_bounding_box**: get minimum box for all annotations in slide

* **manual_bounds**: use given ROI bounds provided by the 'bounds' param

* **polygonal_bounds**: use manually-drawn polygonal (or rectanglar) ROI boundaries

Be sure to checkout the annotations_to_masks_handler.ipynb jupyter notebook for implementation examples.


**Here's a sample result:**

**Before**


.. image:: https://user-images.githubusercontent.com/22067552/63966887-46855c80-ca6a-11e9-8431-932fda6cffc1.png
   :target: https://user-images.githubusercontent.com/22067552/63966887-46855c80-ca6a-11e9-8431-932fda6cffc1.png
   :alt: image


**After**

If the ```polygonal_bounds``` mode is used though its wrapper function (see jupyter), the result is saved mask files named something like: ```TCGA-A2-A0YE_left-59201_top-33493_bottom-63742_right-38093.png```


**Where to look?**

::

   |_ histomicstk/
   |   |_annotations_and_masks/
   |   |  |_annotation_and_mask_utils.py 
   |   |  |_annotations_to_masks_handler.py
   |   |_tests/
   |       |_ annotation_and_mask_utils_test.py
   |       |_ annotations_to_masks_handler_test.py
   |       |_test_files/
   |          |_sample_GTcodes.csv
   |_ docs/
       |_examples/
          |_annotations_to_masks_handler.ipynb


.. automodule:: histomicstk.annotations_and_masks.annotations_to_masks_handler
    :members:
    :undoc-members:
    :show-inheritance:
