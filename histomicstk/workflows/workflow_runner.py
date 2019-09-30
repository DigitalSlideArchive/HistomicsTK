#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 22:09:40 2019

@author: mtageld
"""
from histomicstk.utils.general_utils import Base_HTK_Class

# %% ==========================================================================


class Slide_iterator(Base_HTK_Class):
    """Iterate through large_image items in a girder folder."""

    def __init__(self, gc, source_folder_id, **kwargs):
        """Init Cellularity_Detector_Superpixels object.

        Arguments:
        -----------
        gc : object
            girder client object
        source_folder_id : str
            girder ID of folder in which slides are located
        keep_slides : list
            List of slide names to keep. If None, all are kept.
        discard_slides : list
            List of slide names to discard.
        kwargs : key-value pairs
            The following are already assigned defaults by Base_HTK_Class
            but can be passed here to override defaults
            [verbose, monitorPrefix, logger, logging_savepath,
            suppress_warnings]

        """
        default_attr = {
            'keep_slides': None,
            'discard_slides': [],
        }
        default_attr.update(kwargs)
        super(Slide_iterator, self).__init__(default_attr=default_attr)

        # set attribs
        self.gc = gc
        self.source_folder_id = source_folder_id
        self.set_slide_ids()

    # =========================================================================

    def set_slide_ids(self):
        """Get dict of slide idx, indexed by name."""
        resp = self.gc.get("item?folderId=%s" % self.source_folder_id)
        self.slide_ids = {j['name']: j['_id'] for j in resp}
        # find discard ids
        if self.keep_slides is not None:
            discard = set(self.slide_ids.keys()) - set(self.keep_slides)
            self.discard_slides.extend(list(discard))
            self.discard_slides = list(set(self.discard_slides))
        # only keep what's relevant
        for sn in self.discard_slides:
            del self.slide_ids[sn]

    # =========================================================================

    def run(self):
        """Yields information on one slide at a time."""
        for sname, sid in self.slide_ids.items():
            slide_info = self.gc.get('item/%s/tiles' % sid)
            slide_info['name'] = sname
            slide_info['_id'] = sid
            yield slide_info

# %% ==========================================================================
