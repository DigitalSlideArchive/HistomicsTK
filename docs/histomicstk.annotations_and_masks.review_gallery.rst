Annotation review gallery
==========================

**Overview:**

Feature demo: https://www.youtube.com/watch?v=1wvYV7i7OkQ

This workflow takes a girder folder contains annotated slides, including
special Region-of-Interest (ROI) annotations that enclose annotated regions.
It downloads visualizations of those ROIs and tiles them into a couple of
galleries to be reviewed rapidly by the pathologist. It also adds a link
that the pathologist can use to access the same ROI in the original slide
to pan around it. See the demo video above for details.

**Where to look:**

::

    |_histomicstk/
    |  |_annotations_and_masks/
    |     |_review_gallery.py
    |     |_tests/
    |        |_review_gallery_test.py
    |_ docs/
       |_examples/
          |_review_gallery.ipynb


.. automodule:: histomicstk.annotations_and_masks.review_gallery
    :members:
    :undoc-members:
    :show-inheritance: