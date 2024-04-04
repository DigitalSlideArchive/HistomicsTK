"""
Created on Tue Oct  1 01:38:16 2019.

@author: mtageld
"""
from histomicstk.annotations_and_masks.annotation_and_mask_utils import \
    delete_annotations_in_slide


def cellularity_detection_workflow(
        gc, cdo, slide_id, monitorPrefix='',
        destination_folder_id=None, keep_existing_annotations=False):
    """Run cellularity detection for single slide.

    The cellularity detection algorithm can either be
    Cellularity_detector_superpixels or Cellularity_detector_thresholding.

    Arguments:
    ---------
    gc : object
        girder client object
    cdo : object
        Cellularity_detector object instance. Can either be
        Cellularity_detector_superpixels() or
        Cellularity_detector_thresholding(). The thresholding-based workflow
        seems to be more robust, despite being simpler.
    slide_id : str
        girder id of slide on which workflow is done
    monitorPrefix : str
        this will set the cds monitorPrefix attribute
    destination_folder_id : str or None
        if not None, copy slide to this girder folder and post results
        there instead of original slide.
    keep_existing_annotations : bool
        keep existing annotations in slide when posting results?

    """
    cdo.monitorPrefix = monitorPrefix

    # copy slide to target folder, otherwise work in-place
    if destination_folder_id is not None:
        cdo._print1('%s: copying slide to destination folder' % monitorPrefix)
        resp = gc.post(
            '/item/%s/copy?folderId=%s&copyAnnotations=%s' %
            (slide_id, destination_folder_id, keep_existing_annotations))
        slide_id = resp['_id']

    elif not keep_existing_annotations:
        cdo._print1('%s: deleting existing annotations' % monitorPrefix)
        delete_annotations_in_slide(gc, slide_id)

    # run cds for this slide
    cdo.slide_id = slide_id
    cdo.run()

    return slide_id
