find_package(Cython REQUIRED)
find_package(NumPy REQUIRED)

add_cython_target(_max_clustering_cython CXX)
add_library(_max_clustering_cython MODULE ${_max_clustering_cython})
python_extension_module(_max_clustering_cython)
target_include_directories(_max_clustering_cython PRIVATE ${NumPy_INCLUDE_DIR})

install(TARGETS _max_clustering_cython LIBRARY DESTINATION histomicstk/segmentation/nuclear)
