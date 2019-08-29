# Annotations to masks workflow

**Overview:** 
This includes tools to parse annotations from an item (slide) into masks to use in training and evaluating imaging algorithms. The basic functionality involves mapping annotation rectangles (potentially rotated) and polygons to regions-of-interest (ROIs) and making one mask per region of interest. This extends on some of the workflows described in Amgad et al, 2019:

_Mohamed Amgad, Habiba Elfandy, Hagar Hussein, ..., Jonathan Beezley, Deepak R Chittajallu, David Manthey, David A Gutman, Lee A D Cooper, Structured crowdsourcing enables convolutional segmentation of histology images, Bioinformatics, , btz083, https://doi.org/10.1093/bioinformatics/btz083_

This slide used as a test example:

[TCGA-A2-A0YE-01Z-00-DX1](http://candygram.neurology.emory.edu:8080/histomicstk#?image=5d586d57bd4404c6b1f28640&bounds=53566%2C33193%2C68926%2C40593%2C0)

The user uses a csv file like the one in 
``` ./tests/test_files/sample_GTcodes.csv ```
to control pixel values assigned to mask, overlay order of various annotation groups, which groups are considered to be ROIs, etc.

**Here's the result:**

__Before__

![image](https://user-images.githubusercontent.com/22067552/63966887-46855c80-ca6a-11e9-8431-932fda6cffc1.png)

__After__

Saved mask files:
- TCGA-A2-A0YE-01Z-00-DX1.8A2E3094-5755-42BC-969D-7F0A2ECA0F39_left-59206_top-33505_mag-BASE.png
- TCGA-A2-A0YE-01Z-00-DX1.8A2E3094-5755-42BC-969D-7F0A2ECA0F39_left-58482_top-38217_mag-BASE.png
- TCGA-A2-A0YE-01Z-00-DX1.8A2E3094-5755-42BC-969D-7F0A2ECA0F39_left-57611_top-35827_mag-BASE.png

The code added here handles the following complex situations:
  - Multiple ROIs per item
  - Rotated rectangular annotations and ROIs
  - Polygonal ROIs
  - Overlapping annotations

**Where to look?**

```
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
```
