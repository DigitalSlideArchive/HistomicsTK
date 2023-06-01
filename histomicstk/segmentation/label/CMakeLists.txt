find_package(Cython REQUIRED)
find_package(NumPy REQUIRED)

add_cython_target(_trace_object_boundaries_cython CXX)
add_library(_trace_object_boundaries_cython MODULE ${_trace_object_boundaries_cython})
python_extension_module(_trace_object_boundaries_cython)
message("NUMPY_INCLUDE_DIRS:${NumPy_INCLUDE_DIR}")
target_include_directories(_trace_object_boundaries_cython PRIVATE ${NumPy_INCLUDE_DIR})

install(TARGETS _trace_object_boundaries_cython LIBRARY DESTINATION histomicstk/segmentation/label)
