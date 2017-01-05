#!/bin/bash -e

# Install OpenJPEG, LIBTIFF, and OpenSlide.  These are only intalled if the 
# appropriate environment variable is set.  Here are some sample values
#  OPENJPEG_VERSION  2.1                  2.1.2          1.5.2
#  OPENJPEG_FILE     version.2.1.tar.gz   v2.1.2.tar.gz  version.1.5.2.tar.gz
#  OPENJEPG_DIR      openjpeg-version.2.1 openjpeg-2.1.2 openjpeg-version.1.5.2
#  LIBTIFF_VERSION   4.0.3                4.0.6
#  OPENSLIDE_VERSION 3.4.1
# This also expects the CACHE environment variable to point to a directory that
# can be used for build files.

if [[ -n "$CACHE" && -n "$OPENJPEG_VERSION" && -n "$OPENJPEG_FILE" && -n "$OPENJPEG_DIR" ]]; then
  pushd "$CACHE"
  wget -O "openjpeg-$OPENJPEG_VERSION.tar.gz" "https://github.com/uclouvain/openjpeg/archive/$OPENJPEG_FILE"
  tar -zxf "openjpeg-$OPENJPEG_VERSION.tar.gz"
  cd "$OPENJPEG_DIR"
  cmake .
  make -j 3
  sudo make install
  sudo ldconfig
  popd
fi

if [[ -n "$CACHE" && -n "$LIBTIFF_VERSION" ]]; then
  # Build libtiff so it will use our openjpeg
  pushd "$CACHE"
  wget -O "tiff-$LIBTIFF_VERSION.tar.gz" "http://download.osgeo.org/libtiff/tiff-$LIBTIFF_VERSION.tar.gz"
  tar -zxf "tiff-$LIBTIFF_VERSION.tar.gz"
  cd "tiff-$LIBTIFF_VERSION"
  ./configure
  make -j 3
  sudo make install
  sudo ldconfig
  popd
fi

if [[ -n "$CACHE" && -n "$OPENSLIDE_VERSION" ]]; then
  # Build OpenSlide ourselves so that it will use our libtiff
  pushd "$CACHE"
  wget -O "openslide-$OPENSLIDE_VERSION.tar.gz" "https://github.com/openslide/openslide/archive/v${OPENSLIDE_VERSION}.tar.gz"
  tar -zxf "openslide-$OPENSLIDE_VERSION.tar.gz"
  cd "openslide-$OPENSLIDE_VERSION"
  autoreconf -i
  ./configure
  make -j 3
  sudo make install
  sudo ldconfig
  popd
fi
