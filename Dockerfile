# This is the Dockerfile for the docker image used by all CLIs as a base image
# This image includes a minimal install of ITK

FROM ubuntu:14.04
MAINTAINER Deepak Chittajallu <deepak.chittajallu@kitware.com>

# Install system prerequisites
RUN apt-get update && \
    apt-get install -y git wget python-qt4\
    openslide-tools python-openslide \
    build-essential \
    swig \
    make \
    zlib1g-dev \
    curl \
    libcurl4-openssl-dev \
     libexpat1-dev \
    unzip && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install miniconda
ENV build_path=$PWD/build
RUN mkdir -p $build_path && \
    wget https://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh \
    -O $build_path/install_miniconda.sh && \
    bash $build_path/install_miniconda.sh -b -p $build_path/miniconda && \
    rm $build_path/install_miniconda.sh && \
    chmod -R +r $build_path && \
    chmod +x $build_path/miniconda/bin/python
ENV PATH=$build_path/miniconda/bin:${PATH}

# git clone install ctk-cli
RUN git clone https://github.com/cdeepakroy/ctk-cli.git && cd ctk-cli \
    git checkout 979d8cb671060e787b725b0226332a72a551592e && \
    python setup.py install

# copy HistomicsTK files
ENV htk_path=$PWD/HistomicsTK
RUN mkdir -p $htk_path
COPY . $htk_path/
WORKDIR $htk_path

# Install HistomicsTK and its dependencies
RUN conda config --add channels https://conda.binstar.org/cdeepakroy && \
    conda install --yes pip libgfortran==1.0 openslide-python \
    --file requirements.txt --file requirements_c_conda.txt && \
    pip install -r requirements_c.txt && \
    # Install HistomicsTK
    python setup.py install && \
    # clean up
    conda clean -i -l -t -y && \
    rm -rf /root/.cache/pip/*

#install requirements for itk

WORKDIR /

#need cmake
RUN git clone https://github.com/Kitware/CMake.git
WORKDIR CMake 
#cmake release v3.6.0
RUN git checkout e31084e65745f9dd422c6aff0a2ed4ada6918805
RUN ls
#by default cmake does not use system ssl
RUN ./bootstrap --system-curl && make && make install
  
WORKDIR /
#ITK dependencies


RUN mkdir ninja
WORKDIR ninja

RUN wget https://github.com/ninja-build/ninja/releases/download/v1.7.1/ninja-linux.zip

RUN unzip ninja-linux.zip
RUN ln -s $(pwd)/ninja /usr/bin/ninja

RUN ninja --version

WORKDIR /

RUN git clone https://github.com/InsightSoftwareConsortium/ITK.git
WORKDIR ITK
#need to get the latest tag of master branch in ITK
RUN git describe --abbrev=0 --tags
#get the latest release of ITK
# v4.10.0 = 95291c32dc0162d688b242deea2b059dac58754a
RUN git checkout $(git describe --abbrev=0 --tags)
WORKDIR /
RUN mkdir ITKbuild
WORKDIR ITKbuild

RUN cmake \
-G Ninja \
-DBUILD_EXAMPLES:BOOL=OFF \
-DBUILD_TESTING:BOOL=OFF \
-DBUILD_SHARED_LIBS:BOOL=ON \
-DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON \
-DITK_LEGACY_REMOVE:BOOL=ON \
-DITK_BUILD_DEFAULT_MODULES:BOOL=OFF \
-DITK_USE_SYSTEM_LIBRARIES:BOOL=ON \
-DModule_ITKIOImageBase:BOOL=ON \
-DModule_ITKTestKernel:BOOL=ON \
-DCMAKE_INSTALL_PREFIX:PATH=/build/miniconda \
-DITK_WRAP_PYTHON=ON \
-DPYTHON_INCLUDE_DIR:FILEPATH=/build/miniconda/include/python2.7 \
-DPYTHON_LIBRARY:FILEPATH=/build/miniconda/lib/libpython2.7.so \
-DPYTHON_EXECUTABLE:FILEPATH=/build/miniconda/bin/python \
-DBUILD_EXAMPLES:BOOL=OFF -DBUILD_TESTING:BOOL=OFF ../ITK && ninja

RUN ninja install

#delete build files
RUN rm -rf /ITKbuild

# pregenerate font cache
RUN python -c "from matplotlib import pylab"

# define entrypoint through which all CLIs can be run
WORKDIR $htk_path/server
ENTRYPOINT ["/build/miniconda/bin/python", "cli_list_entrypoint.py"]
