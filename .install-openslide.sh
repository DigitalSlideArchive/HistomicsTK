#!/bin/bash

PREFIX="$CACHE/openslide-$OPENSLIDE_VERSION"
if [[ ! -f "$PREFIX/bin/openslide-show-properties" || -n "$UPDATE_CACHE" ]] ; then
  rm -fr "$PREFIX"
  mkdir -p "$PREFIX/src"
  curl -L "https://github.com/openslide/openslide/releases/download/v${OPENSLIDE_VERSION}/openslide-${OPENSLIDE_VERSION}.tar.gz" | gunzip -c | tar -x -C "$PREFIX/src" --strip-components 1
  pushd "$PREFIX/src"
  ./configure "--prefix=$PREFIX"
  make install
  popd
fi
export PATH="$PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$PREFIX/lib"
