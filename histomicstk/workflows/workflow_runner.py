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


class Workflow_runner(Base_HTK_Class):
    """Run workflow for all slides in a girder folder."""

    def __init__(
            self, slide_iterator, workflow, workflow_kwargs,
            recursive=False, catch_exceptions=True, **kwargs):
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
        recursive : bool
            whether to run the workflow recursively on all subfolders
        catch_exceptions : bool
            whether to catch exceptions. You may want to set to false if
            for example you want to run with a debugger
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
        self.recursive = recursive
        self.catch_exceptions = catch_exceptions
        if self.keep_log:
            self.exception_path = self.logname.replace(
                '.log', '_EXCEPTIONS.log')
        self.slide_iterator = slide_iterator
        self.gc = self.slide_iterator.gc
        self.si = slide_iterator.run()
        self.originalPrefix = self.monitorPrefix

    # =========================================================================

    def run(self):
        """Run workflow for all slides."""
        self.n_slides = len(self.slide_iterator.slide_ids)

        def _run_slide(self, monitorStr):

            slide_info = next(self.si)
            monitorStr += " (%s)" % (slide_info['name'])

            _ = self.workflow(
                slide_id=slide_info['_id'], monitorPrefix=monitorStr,
                **self.workflow_kwargs)

        for sno in range(self.n_slides):

            monitorStr = "%s: slide %d of %d" % (
                self.monitorPrefix, sno + 1, self.n_slides)

            if not self.catch_exceptions:
                _run_slide(self, monitorStr)
            else:
                try:
                    _run_slide(self, monitorStr)

                except Exception as e:

                    if self.keep_log:
                        self.cpr1.logger.exception(
                            "%s: SEE EXCEPTIONS FILE: %s" % (
                                monitorStr, self.exception_path))
                        with open(self.exception_path, 'a') as f:
                            print(str(e))
                            f.write("%s\n" % monitorStr)
                            f.write(e.__repr__())
                            f.write("\n---------------------------------\n")
                    else:
                        print(e.__repr__())

        if self.recursive:
            # for each subfolder, call self
            for folder in self.gc.listFolder(
                    parentId=self.slide_iterator.source_folder_id):

                fpath = self.gc.get('/folder/%s/rootpath' % folder['_id'])
                fpath = "/".join(
                    [j['object']['name'] for j in fpath]
                ) + "/" + folder['name'] + "/"

                self.monitorPrefix = "%s: %s" % (self.originalPrefix, fpath)

                # update slide iterator for subfolder
                self.slide_iterator.source_folder_id = folder['_id']
                self.slide_iterator.set_slide_ids()
                self.si = self.slide_iterator.run()

                # recurse
                self.run()

# %% ==========================================================================


class Annotation_iterator(Base_HTK_Class):
    """Iterate through annotations in a girder item (slide)."""

    def __init__(
            self, gc, slide_id, callback=None, callback_kwargs=None, **kwargs):
        """Init Annotation_iterator object.

        Arguments
        -----------
        gc : object
            girder client object
        slide_id : str
            girder ID of slide (item)
        callback : function
            function to apply to each annotation. Must accept at least
            the parameters "gc" and "annotation" and these will be passed
            internally to it.
        callback_kwargs : dict
            kwargs to pass to the callback (other than gc and annotation)
        kwargs : key-value pairs
            The following are already assigned defaults by Base_HTK_Class
            but can be passed here to override defaults
            [verbose, monitorPrefix, logger, logging_savepath,
            suppress_warnings]

        """
        default_attr = dict()
        default_attr.update(kwargs)
        super(Annotation_iterator, self).__init__(default_attr=default_attr)

        # set attribs
        self.gc = gc
        self.slide_id = slide_id
        self.callback = callback
        self.callback_kwargs = callback_kwargs

        # get annotations for slide
        self.slide_annotations = self.gc.get(
            '/annotation/item/' + self.slide_id)
        self.n_annotations = len(self.slide_annotations)

    def yield_callback_output_for_annotation(self):
        """Yield callback output for one annotation at a time."""
        # yield one annotation at a time
        for annidx, ann in enumerate(self.slide_annotations):
            if self.verbose > 0:
                print("%s: annotation %d of %d" % (
                    self.monitorPrefix, annidx + 1, self.n_annotations))
            if self.callback is None:
                yield ann
            else:
                yield self.callback(
                    gc=self.gc, annotation=ann, **self.callback_kwargs)

    def apply_callback_to_all_annotations(self):
        """Apply callback to all annotations and resturn output list."""
        runner = self.yield_callback_output_for_annotation()
        outputs = []
        for annidx in range(self.n_annotations):
            outputs.append(next(runner))
        return outputs

# %% ==========================================================================
