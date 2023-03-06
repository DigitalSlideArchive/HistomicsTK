# This Dockerfile is used to generate the docker image dsarchive/histomicstk
# This docker image includes the HistomicsTK python package along with its
# dependencies.
#
# All plugins of HistomicsTK should derive from this docker image

FROM python:3.8-slim
LABEL maintainer="Kitware, Inc. <kitware@kitware.com>"

RUN apt-get update && \
    apt-get install --yes --no-install-recommends \
    # For installing pip \
    curl \
    ca-certificates \
    # For versioning \
    git \
    # for convenience \
    wget \
    # Needed for building \
    build-essential \
    # can speed up large_image caching \
    memcached \
    && \
    # Clean up to reduce docker size \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Make a specific version of python the default and install pip
# RUN rm -f /usr/bin/python && \
#     rm -f /usr/bin/python3 && \
#     ln `which python3.8` /usr/bin/python && \
#     ln `which python3.8` /usr/bin/python3 && \
#     curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
#     python get-pip.py && \
#     rm get-pip.py && \
#     ln `which pip3` /usr/bin/pip && \
#     python --version

# copy HistomicsTK files
ENV htk_path=$PWD/HistomicsTK
RUN mkdir -p $htk_path

RUN pip install --no-cache-dir --upgrade pip setuptools && \
    # Install bokeh to help debug dask \
    pip install --no-cache-dir 'bokeh>=0.12.14,<3' && \
    # Install large_image memcached and sources extras \
    pip install --no-cache-dir 'large-image[all]' --find-links https://girder.github.io/large_image_wheels && \
    # Install girder-client \
    pip install --no-cache-dir girder-client && \
    # Install some other dependencies here to save time in the histomicstk \
    # install step \
    pip install --no-cache-dir nimfa numpy scipy Pillow pandas scikit-image scikit-learn imageio 'shapely[vectorized]' opencv-python-headless sqlalchemy matplotlib 'dask[dataframe]' distributed && \
    # clean up \
    rm -rf /root/.cache/pip/*

# Install the latest version of large_image.  This can be disabled if the
# latest version we need has had an official release
# RUN cd /opt && \
#     git clone https://github.com/girder/large_image && \
#     cd large_image && \
#     # git checkout some-branch && \
#     pip install .[all] -r requirements-dev.txt --find-links https://girder.github.io/large_image_wheels

COPY . $htk_path/
COPY histomicstk/cli/dask_config.yaml /root/.config/dask/dask_config.yaml
WORKDIR $htk_path

# Install HistomicsTK and its dependencies
RUN pip install --no-cache-dir . --find-links https://girder.github.io/large_image_wheels && \
    pip install --no-cache-dir virtualenv && \
    rm -rf /root/.cache/pip/*

# Show what was installed
RUN pip freeze

# pregenerate font cache
RUN python -c "from matplotlib import pylab"

# define entrypoint through which all CLIs can be run
WORKDIR $htk_path/histomicstk/cli

# Test our entrypoint.  If we have incompatible versions of numpy and
# openslide, one of these will fail
RUN python -m slicer_cli_web.cli_list_entrypoint --list_cli
RUN python -m slicer_cli_web.cli_list_entrypoint ColorDeconvolution --help
# Debug import time
RUN python -X importtime ColorDeconvolution/ColorDeconvolution.py --help

ENV PYTHONUNBUFFERED=TRUE

ENTRYPOINT ["/bin/bash", "docker-entrypoint.sh"]
