# This is the Dockerfile for the docker image used by all CLIs as a base image

FROM: ubuntu:14.04
MAINTAINER Deepak Chittajallu <deepak.chittajallu@kitware.com>

# Install system prerequisites
RUN apt-get update
RUN apt-get install -y git wget

# define a few paths
RUN build_path=$PWD/build
RUN mkdir -p $build_path

# Install miniconda
RUN wget -O $build_path/install_miniconda.sh \
    https://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh
RUN bash $build_path/install_miniconda.sh -b -p $build_path/miniconda
RUN source $build_path/miniconda/bin/activate $build_path/miniconda
RUN conda update --yes --all

# git clone HistomicsTK (toolkit with some core common functionality)
RUN git clone git@github.com:DigitalSlideArchive/HistomicsTK.git
RUN git -C HistomicsTK checkout 62c988a4084104bb62b77d6c82247dd32dfc7d6f

# Install dependencies of HistomicsTK
RUN conda config --add channels https://conda.binstar.org/cdeepakroy
RUN conda install conda install --yes \
    libgfortran==1.0 numpy==1.10.2 scipy==0.16.0 scikit-image==0.12.3 \
RUN conda install --yes \
    --file HistomicsTK/requirements.txt \
    --file HistomicsTK/requirements_c.txt \
    setuptools==19.4

# Install HistomicsTK
RUN cd HistomicsTK
RUN python setup.py install
RUN cd ..


