# Tissue Detection module

**Overview:** 
This includes tools to detect tissue from an item (slide) using its thumbnail. The basic functionality includes a series of gaussian smoothing and otsu thresholding steps to detect background versus foreground pixels. Optionally, an initial step is performed whereby color deconvolution is used to deparate hematoxylin and eosin stains (assuming H&E stained slides) to make sure only cellular areas are segmented. This proves to be useful in getting rid of sharpie markers. A size threshold is used to keep only largest contiguous tissue regions.

This slide used as a test example:

[TCGA-A1-A0SK-01Z-00-DX1](http://candygram.neurology.emory.edu:8080/histomicstk#?image=5d817f5abd4404c6b1f744bb)

**@TODO** -> Color normalization prior to deconvolution

**Here's the result:**

From left to right: Slide thumbnail, Tissue regions, Largest tissue region

![image](https://user-images.githubusercontent.com/22067552/65110899-f4b85e00-d9a7-11e9-8782-5a5f18992ae4.png)

**Where to look?**

```
|_ histomicstk/
|   |_saliency/
|      |_tissue_detection.py 
|      |_tests/
|         |_tissue_detection_test.py
|_ docs/
    |_examples/
       |_tissue_detection.ipynb
```
