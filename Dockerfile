# This is the Dockerfile for the docker image used by all CLIs as a base image

FROM ubuntu:14.04
MAINTAINER Deepak Chittajallu <deepak.chittajallu@kitware.com>

# Install system prerequisites
RUN apt-get update && apt-get install -y git wget

# define a few paths
ENV build_path=$PWD/build
RUN mkdir -p $build_path

# Install miniconda
RUN wget -O $build_path/install_miniconda.sh \
    https://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh && \
    bash $build_path/install_miniconda.sh -b -p $build_path/miniconda
ENV PATH=$build_path/miniconda/bin:${PATH}
RUN conda update --yes --all

# git clone HistomicsTK (toolkit with some core common functionality)
RUN git clone https://github.com/DigitalSlideArchive/HistomicsTK.git
RUN git -C HistomicsTK checkout 2b004cca07f98b34cad054ec0de3f1e43cb219cf

# Install dependencies of HistomicsTK
RUN conda config --add channels https://conda.binstar.org/cdeepakroy
RUN conda install --yes \
    libgfortran==1.0 numpy==1.10.2 scipy==0.16.0 \
    scikit-image==0.12.3 scikit-learn==0.17.1
RUN conda install --yes --file HistomicsTK/requirements.txt
RUN pip install -r HistomicsTK/requirements_c.txt

# Install HistomicsTK
RUN cd HistomicsTK && python setup.py install