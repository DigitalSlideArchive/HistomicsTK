# This Dockerfile is used to generate the docker image dsarchive/histomicstk
# This docker image includes the HistomicsTK python package along with its
# dependencies.
#
# All plugins of HistomicsTK should derive from this docker image

FROM dsarchive/base_docker_image
MAINTAINER Deepak Chittajallu <deepak.chittajallu@kitware.com>

# copy HistomicsTK files
ENV htk_path=$PWD/HistomicsTK
RUN mkdir -p $htk_path

RUN apt-get update && \
    apt-get install -y --no-install-recommends memcached && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
COPY . $htk_path/
WORKDIR $htk_path

# Install HistomicsTK and its dependencies
#   Upgrade setuptools, as the version in Conda won't upgrade cleanly unless it
# is ignored.
RUN pip install --no-cache-dir --upgrade --ignore-installed pip setuptools && \
    pip install --no-cache-dir 'bokeh>=0.12.14' && \
    # Install large_image
    pip install --no-cache-dir 'large-image[memcached,openslide,tiff,pil]' -f https://manthey.github.io/large_image_wheels && \
    pip install --no-cache-dir scikit-build setuptools-scm Cython cmake && \
    # Install HistomicsTK
    pip install . && \
    # Create separate virtual environments with CPU and GPU versions of tensorflow
    pip install --no-cache-dir 'virtualenv<16.4.0' && \
    virtualenv --system-site-packages /venv-gpu && \
    chmod +x /venv-gpu/bin/activate && \
    /venv-gpu/bin/pip install --no-cache-dir 'tensorflow-gpu>=1.3.0' && \
    # clean up
    rm -rf /root/.cache/pip/*

# git clone install slicer_cli_web
RUN mkdir -p /build && \
    cd /build && \
    git clone https://github.com/girder/slicer_cli_web.git

# Show what was installed
RUN pip freeze

# pregenerate font cache
RUN python -c "from matplotlib import pylab"

# pregenerate libtiff wrapper.  This also tests libtiff for failures
RUN python -c "import libtiff"

# define entrypoint through which all CLIs can be run
WORKDIR $htk_path/histomicstk/cli

# Test our entrypoint.  If we have incompatible versions of numpy and
# openslide, one of these will fail
RUN python /build/slicer_cli_web/server/cli_list_entrypoint.py --list_cli
RUN python /build/slicer_cli_web/server/cli_list_entrypoint.py ColorDeconvolution --help

ENTRYPOINT ["/bin/bash", "docker-entrypoint.sh"]
