FROM dockcross/manylinux-x64

# Don't build python 3.4 wheels.
RUN rm -r /opt/python/cp27-cp27m
RUN rm -r /opt/python/cp34*
RUN rm -r /opt/python/cp38*

RUN for PYBIN in /opt/python/*/bin; do \
        ${PYBIN}/pip install --upgrade pip; \
    done

RUN for PYBIN in /opt/python/*/bin; do \
        ${PYBIN}/pip install libtiff openslide_python pyvips -f https://manthey.github.io/large_image_wheels large_image_source_tiff large_image_source_openslide large_image_source_pil; \
    done

RUN for PYBIN in /opt/python/*/bin; do \
        ${PYBIN}/pip install Cython setuptools-scm; \
    done

ENV htk_path=/HistomicsTK
RUN mkdir -p $htk_path

COPY . $htk_path/

ARG CIRCLE_BRANCH
ENV CIRCLE_BRANCH=$CIRCLE_BRANCH

RUN cd $htk_path && \
    for PYBIN in /opt/python/*/bin; do \
        ${PYBIN}/pip install . && \
        ${PYBIN}/pip wheel . --no-deps -w /io/wheelhouse/; \
    done && \
    for WHL in /io/wheelhouse/histomicstk*.whl; do \
      auditwheel repair "${WHL}" -w /io/wheelhouse/; \
    done && \
    ls -l /io/wheelhouse && \
    mkdir /io/wheels && \
    cp /io/wheelhouse/histomicstk*many* /io/wheels/. 
