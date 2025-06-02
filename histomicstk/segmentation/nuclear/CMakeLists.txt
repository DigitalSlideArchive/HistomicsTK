find_package(Python COMPONENTS Interpreter Development.Module NumPy REQUIRED)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/_max_clustering_cython.cpp
  COMMENT
    "Making ${CMAKE_CURRENT_BINARY_DIR}/_max_clustering_cython.cpp from ${CMAKE_CURRENT_SOURCE_DIR}/_max_clustering_cython.pyx"
  COMMAND Python::Interpreter -m cython --cplus -o "${CMAKE_CURRENT_BINARY_DIR}/_max_clustering_cython.cpp"
          "${CMAKE_CURRENT_SOURCE_DIR}/_max_clustering_cython.pyx"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/_max_clustering_cython.pyx"
  VERBATIM)

python_add_library(_max_clustering_cython MODULE
                   "${CMAKE_CURRENT_BINARY_DIR}/_max_clustering_cython.cpp"
                   WITH_SOABI)

target_link_libraries(_max_clustering_cython PRIVATE Python::Module Python::NumPy)

install(TARGETS _max_clustering_cython DESTINATION histomicstk/segmentation/nuclear)
