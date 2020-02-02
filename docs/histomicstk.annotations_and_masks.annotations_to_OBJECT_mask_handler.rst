Annotations to OBJECT masks workflow
======================================

**Overview:**

This includes tools to parse annotations from an item (slide) into masks to use in training and evaluating object localization, classification, and segmentation imaging algorithms (eg. MASK-RCNN). Two "versions" of this workflow exist:

* Get labeled mask for any region in a whole-slide image (user-defined)

* Get labeled mask for areas enclosed within special "region-of-interest" (ROI) annotations that have been drawn by the user. This involves mapping annotations (rectangles/polygons) to ROIs and making one mask per ROI.

This extends on some of the workflows described in Amgad et al, 2019:

*Mohamed Amgad, Habiba Elfandy, Hagar Hussein, ..., Jonathan Beezley, Deepak R Chittajallu, David Manthey, David A Gutman, Lee A D Cooper, Structured crowdsourcing enables convolutional segmentation of histology images, Bioinformatics, , btz083, https://doi.org/10.1093/bioinformatics/btz083*

This slide used as a test example:

`TCGA-A2-A0YE-01Z-00-DX1 <http://candygram.neurology.emory.edu:8080/histomicstk#?image=5d586d57bd4404c6b1f28640&bounds=53566%2C33193%2C68926%2C40593%2C0>`_

The user uses a csv file like the one in
```histomicstk/annotations_and_masks/tests/test_files/sample_GTcodes.csv ```
to control pixel values assigned to mask, overlay order of various annotation groups, which groups are considered to be ROIs, etc. Note that we use the girder definition of term "group" here, which is an annotation style indicating a certain class, such as "tumor" or "necrosis".

**What is the difference between this and annotations_to_masks_handler?!**

The difference between this and version 1, found at
```histomicstk.annotations_and_masks.annotations_to_masks_handler```
is that this (version 2) gets the contours first, including cropping to wanted ROI boundaries and other processing using shapely, and THEN parses these into masks. This enables us to differentiate various objects to use the data for object localization/classification/segmentation tasks. If you would like to get semantic segmentation masks, i.e. you do not really care about individual objects, you can use either version 1 or this handler using the semantic run mode. They re-use much of the same code-base, but some edge cases maybe better handled by version 1. For example, since this version uses shapely first to crop, some objects may be incorrectly parsed by shapely. Version 1, using PIL.ImageDraw may not have these problems.

Bottom line is: if you need semantic segmentation masks, it is probably safer to use version 1 (annotations to masks handler), whereas if you need object segmentation masks, this handler should be used in object run mode.

Be sure to checkout the annotations_to_OBJECT_mask_handler.ipynb jupyter notebook for implementation examples.


**Where to look?**

::

   |_ histomicstk/
   |   |_annotations_and_masks/
   |   |  |_annotation_and_mask_utils.py
   |   |  |_annotations_to_object_mask_handler.py
   |   |_tests/
   |       |_ annotation_and_mask_utils_test.py
   |       |_ annotations_to_object_mask_handler_test.py
   |       |_test_files/
   |          |_sample_GTcodes.csv
   |_ docs/
       |_examples/
          |_annotations_to_OBJECT_mask_handler.ipynb


.. automodule:: histomicstk.annotations_and_masks.annotations_to_object_mask_handler
    :members:
    :undoc-members:
    :show-inheritance: