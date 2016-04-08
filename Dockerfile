# This is the Dockerfile for the docker image used by all CLIs as a base image

FROM ubuntu:14.04
MAINTAINER Deepak Chittajallu <deepak.chittajallu@kitware.com>

# Install system prerequisites
RUN apt-get update && \
    apt-get install -y git wget python-qt4\
    openslide-tools python-openslide && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# define a few paths
ENV build_path=$PWD/build
RUN mkdir -p $build_path

# Install miniconda
RUN wget https://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh \
    -O $build_path/install_miniconda.sh && \
    bash $build_path/install_miniconda.sh -b -p $build_path/miniconda && \
    rm $build_path/install_miniconda.sh
ENV PATH=$build_path/miniconda/bin:${PATH}

# git clone HistomicsTK, checkout a branch/release, and install it
RUN git clone https://github.com/DigitalSlideArchive/HistomicsTK.git && \
    cd HistomicsTK && git checkout AutoCreateRESTEndPointForSlicerCLI && \
    # Install dependencies
    conda config --add channels https://conda.binstar.org/cdeepakroy && \
    conda install --yes libgfortran==1.0 openslide-python \
    --file requirements.txt --file requirements_c_conda.txt && \
    pip install -r requirements_c.txt && \
    # Install HistomicsTK
    python setup.py install && \
    # clean up
    conda clean -i -l -t -y && \
    rm -rf /root/.cache/pip/*

