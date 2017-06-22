# This Dockerfile is used to generate the docker image dsarchive/histomicstk
# This docker image includes miniconda, python wrapped ITK, and the HistomicsTK
# python package along with their dependencies.
#
# All plugins of HistomicsTK should derive from this docker image

FROM dsarchive/base_docker_image
MAINTAINER Deepak Chittajallu <deepak.chittajallu@kitware.com>

# copy HistomicsTK files
ENV htk_path=$PWD/HistomicsTK
RUN mkdir -p $htk_path
COPY . $htk_path/
WORKDIR $htk_path

# Install HistomicsTK and its dependencies
#   Upgrade setuptools, as the version in Conda won't upgrade cleanly unless it
# is ignored.
RUN pip install --upgrade --ignore-installed pip setuptools && \
    pip install --upgrade 'git+https://github.com/cdeepakroy/ctk-cli' && \
    # Install requirements.txt via pip; installing via conda causes
    # version issues with our home-built libtif.
    # Try twice; conda sometimes causes pip to fail the first time, but if it
    # fails twice then there is a real issue.
    pip install -r requirements.txt && \
    # Install large_image
    pip install 'git+https://github.com/girder/large_image#egg=large_image' && \
    # Ensure we have a locally built Pillow and openslide in conda's environment
    pip install --upgrade --no-cache-dir --force-reinstall --ignore-installed \
      openslide-python \
      Pillow && \
    # Ensure we have the latest libtif
    pip install --force-reinstall --ignore-installed --upgrade 'git+https://github.com/pearu/pylibtiff@33735eb7197eb33ed1a50bbc4bc5a47ce4305e92' && \
    # Install HistomicsTK
    python setup.py install && \
    # clean up
    conda clean -i -l -t -y && \
    rm -rf /root/.cache/pip/*

# git clone install slicer_cli_web
RUN cd /build && \
    git clone https://github.com/girder/slicer_cli_web.git

# Show what was installed
RUN conda list

# pregenerate font cache
RUN python -c "from matplotlib import pylab"

# pregenerate libtiff wrapper.  This also tests libtiff for failures
RUN python -c "import libtiff"

# define entrypoint through which all CLIs can be run
WORKDIR $htk_path/server

# Test our entrypoint.  If we have incompatible versions of numpy and
# openslide, one of these will fail
RUN python /build/slicer_cli_web/server/cli_list_entrypoint.py --list_cli
RUN python /build/slicer_cli_web/server/cli_list_entrypoint.py ColorDeconvolution --help

ENTRYPOINT ["/build/miniconda/bin/python", "/build/slicer_cli_web/server/cli_list_entrypoint.py"]
