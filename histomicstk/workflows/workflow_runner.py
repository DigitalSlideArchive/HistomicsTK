#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 22:09:40 2019.

@author: mtageld
"""
from histomicstk.utils.general_utils import Base_HTK_Class

# %% ==========================================================================


class Slide_iterator(Base_HTK_Class):
    """Iterate through large_image items in a girder folder."""

    def __init__(self, gc, source_folder_id, **kwargs):
        """Init Slide_iterator object.

        Arguments
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
        resp = self.gc.get(
            "item?folderId=%s&limit=1000000" % self.source_folder_id)
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
        """Yield information on one slide at a time."""
        for sname, sid in self.slide_ids.items():
            try:
                slide_info = self.gc.get('item/%s/tiles' % sid)
            except Exception as e:
                print(str(e))
                slide_info = dict()
            slide_info['name'] = sname
            slide_info['_id'] = sid
            yield slide_info

# %% ==========================================================================
# =============================================================================


class Workflow_runner(Base_HTK_Class):
    """Run workflow for all slides in a girder folder."""

    def __init__(self, slide_iterator, workflow, workflow_kwargs, **kwargs):
        """Init Workflow_runner object.

        Arguments
        -----------
        slide_iterator : object
            Slide_iterator object
        workflow : method
            method whose parameters include slide_id and monitorPrefix,
            which is called for each slide
        workflow_kwargs : dict
            keyword arguments for the workflow method
        kwargs : key-value pairs
            The following are already assigned defaults by Base_HTK_Class
            but can be passed here to override defaults
            [verbose, monitorPrefix, logging_savepath, suppress_warnings]

        """
        default_attr = dict()
        default_attr.update(kwargs)
        super(Workflow_runner, self).__init__(default_attr=default_attr)

        # set attribs
        self.workflow = workflow
        self.workflow_kwargs = workflow_kwargs
        self.exception_path = self.logname.replace('.log', '_EXCEPTIONS.log')
        self.slide_iterator = slide_iterator
        self.si = slide_iterator.run()

    # =========================================================================

    def run(self):
        """Run workflow for all slides."""
        self.n_slides = len(self.slide_iterator.slide_ids)

        for sno in range(self.n_slides):

            monitorStr = "%s: slide %d of %d" % (
                self.monitorPrefix, sno + 1, self.n_slides)

            try:
                slide_info = next(self.si)
                monitorStr += " (%s)" % (slide_info['name'])

                _ = self.workflow(
                    slide_id=slide_info['_id'], monitorPrefix=monitorStr,
                    **self.workflow_kwargs)

            except Exception as e:
                self.cpr1.logger.exception("%s: SEE EXCEPTIONS FILE: %s" % (
                    monitorStr, self.exception_path))
                with open(self.exception_path, 'a') as f:
                    print(e)
                    f.write("%s\n" % monitorStr)
                    f.write(str(e))
                    f.write("\n---------------------------------\n")


# %% ==========================================================================
# =============================================================================
