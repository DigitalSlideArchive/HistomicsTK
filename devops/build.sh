#!/bin/bash

docker exec -it histomicstk_histomicstk bash -c "
     cd /opt/histomicstk/build && \
     cmake -DRUN_CORE_TESTS:BOOL=OFF \
           -DGIRDER_EXTERNAL_DATA_STORE:PATH=/data \
           -DTEST_PLUGINS:STRING=HistomicsTK \
           ../girder/ && \
     make "
