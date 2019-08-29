# Masks to annotations workflow

**Overview:**

This contains the reverse functionality to ``annotations_to_masks.py``. Namely, this includes two key functionalities:

- Parsing annotatoins from a groung truth mask into contours (pandas dataframe)

- Parsing contours into a format that is compatible with large_image annotation schema and
ready to push for visualization in Digital Slide Archive HistomicsTK. 


This extends on some of the workflows described in Amgad et al, 2019:

_Mohamed Amgad, Habiba Elfandy, Hagar Hussein, ..., Jonathan Beezley, Deepak R Chittajallu, David Manthey, David A Gutman, Lee A D Cooper, Structured crowdsourcing enables convolutional segmentation of histology images, Bioinformatics, 2019, btz083_


This slide used as a test example:

[TCGA-A2-A0YE-01Z-00-DX1](http://candygram.neurology.emory.edu:8080/histomicstk#?image=5d586d76bd4404c6b1f286ae&bounds=54743%2C32609%2C68828%2C39395%2C0 )


**Where to look?**

```
|_ histomicstk/
|   |_annotations_and_masks/
|   |   |_masks_to_annotations_handler.py 
|   |_tests/
|       |_masks_to_annotations_handler_test.py
|       |_test_files/
|          |_sample_GTcodes.csv 
|          |_sample_contours_df.tsv 
|          |_TCGA-A2-A0YE-01Z-00-DX1.8A2E3094-5755-42BC-969D-7F0A2ECA0F39_left-59206_top-33505_mag-BASE.png
|_ docs/
    |_examples/
       |_ masks_to_annotations_handler.ipynb
```
