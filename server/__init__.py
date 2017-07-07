import os

from girder import events
from girder.utility.webroot import Webroot

from .handlers import process_annotations

_template = os.path.join(
    os.path.dirname(__file__),
    'webroot.mako'
)


def load(info):

    girderRoot = info['serverRoot']
    histomicsRoot = Webroot(_template)
    histomicsRoot.updateHtmlVars(girderRoot.vars)
    histomicsRoot.updateHtmlVars({'title': 'HistomicsTK'})

    info['serverRoot'].histomicstk = histomicsRoot
    info['serverRoot'].girder = girderRoot

    # auto-ingest annotations into database when a .anot file is uploaded
    events.bind('data.process', 'HistomicsTK', process_annotations)
