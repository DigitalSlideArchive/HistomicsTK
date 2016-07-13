FROM ubuntu:14.04
MAINTAINER Bilal Salam <bilal.salam@kitware.com>


RUN apt-get update && \
apt-get install -y \
git \
wget \
python-qt4 \
openslide-tools python-openslide \
build-essential \
swig \
make \
zlib1g-dev \
curl \
libcurl4-openssl-dev \
libexpat1-dev \
unzip \
libhdf5-dev \
libjpeg-dev \
libopenslide-dev \
libpng12-dev \
libpython3-dev \
libtiff5-dev \
cmake \
openslide-tools && \
apt-get autoremove && \
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /

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


ENV htk_path=$PWD/HistomicsTK
RUN mkdir -p $htk_path



WORKDIR /

#ITK dependencies

RUN mkdir ninja && \
cd ninja && \
wget https://github.com/ninja-build/ninja/releases/download/v1.7.1/ninja-linux.zip && \
unzip ninja-linux.zip && \
ln -s $(pwd)/ninja /usr/bin/ninja  


WORKDIR /
#need to get the latest tag of master branch in ITK
# v4.10.0 = 95291c32dc0162d688b242deea2b059dac58754a
RUN git clone https://github.com/InsightSoftwareConsortium/ITK.git && \
cd ITK && \
git checkout $(git describe --abbrev=0 --tags) && \
#now get openslide
cd Modules/External && \
git clone https://github.com/InsightSoftwareConsortium/ITKIOOpenSlide.git && \
cd / && \
mkdir ITKbuild && \
cd ITKbuild && \
cmake \
-G Ninja \
-DITK_USE_SYSTEM_SWIG:BOOL=OFF \
-DBUILD_EXAMPLES:BOOL=OFF \
-DBUILD_TESTING:BOOL=OFF \
-DBUILD_SHARED_LIBS:BOOL=ON \
-DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON \
-DITK_LEGACY_REMOVE:BOOL=ON \
-DITK_BUILD_DEFAULT_MODULES:BOOL=ON \
-DITK_USE_SYSTEM_LIBRARIES:BOOL=ON \
-DModule_ITKIOImageBase:BOOL=ON \
-DModule_ITKSmoothing:BOOL=ON \
-DModule_ITKTestKernel:BOOL=ON \
-DModule_IOOpenSlide:BOOL=ON \
-DCMAKE_INSTALL_PREFIX:PATH=/build/miniconda \
-DITK_WRAP_PYTHON=ON \
-DPYTHON_INCLUDE_DIR:FILEPATH=/build/miniconda/include/python2.7 \
-DPYTHON_LIBRARY:FILEPATH=/build/miniconda/lib/libpython2.7.so \
-DPYTHON_EXECUTABLE:FILEPATH=/build/miniconda/bin/python \
-DBUILD_EXAMPLES:BOOL=OFF -DBUILD_TESTING:BOOL=OFF ../ITK && \
ninja  && \
ninja install 

rm -rf ITK ITKbuild 

WORKDIR $htk_path



