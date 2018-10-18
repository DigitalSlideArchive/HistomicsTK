#!/bin/bash

docker exec -it histomicstk_histomicstk bash -c "
  sudo girder-install plugin -s /opt/histomicstk/HistomicsTK"
