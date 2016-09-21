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
RUN conda config --add channels https://conda.binstar.org/cdeepakroy && \
    conda install --yes pip libgfortran==1.0 openslide-python \
    --file requirements_c_conda.txt && \
    pip install -r requirements.txt -r requirements_c.txt && \
    # Install HistomicsTK
    python setup.py install && \
    # clean up
    conda clean -i -l -t -y && \
    rm -rf /root/.cache/pip/*

RUN conda install --yes pip libgfortran==1.0 setuptools==19.4 ctk-cli==1.3.1 \
    openslide-python --file requirements_c_conda.txt && \
    pip install -r requirements.txt -r requirements_c.txt && \
    # Install HistomicsTK
    python setup.py install && \
    # clean up
    conda clean -i -l -t -y && \
    rm -rf /root/.cache/pip/*

# pregenerate font cache
RUN python -c "from matplotlib import pylab"

# git clone install slicer_cli_web
RUN cd /build && git clone https://github.com/girder/slicer_cli_web.git

# define entrypoint through which all CLIs can be run
WORKDIR $htk_path/server
ENTRYPOINT ["/build/miniconda/bin/python", "/build/slicer_cli_web/server/cli_list_entrypoint.py"]
