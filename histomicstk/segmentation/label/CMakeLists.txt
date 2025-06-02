find_package(Python COMPONENTS Interpreter Development.Module NumPy REQUIRED)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/_trace_object_boundaries_cython.cpp
  COMMENT
    "Making ${CMAKE_CURRENT_BINARY_DIR}/_trace_object_boundaries_cython.cpp from ${CMAKE_CURRENT_SOURCE_DIR}/_trace_object_boundaries_cython.pyx"
  COMMAND Python::Interpreter -m cython --cplus -o "${CMAKE_CURRENT_BINARY_DIR}/_trace_object_boundaries_cython.cpp"
          "${CMAKE_CURRENT_SOURCE_DIR}/_trace_object_boundaries_cython.pyx"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/_trace_object_boundaries_cython.pyx"
  VERBATIM)

python_add_library(_trace_object_boundaries_cython MODULE
                   "${CMAKE_CURRENT_BINARY_DIR}/_trace_object_boundaries_cython.cpp"
                   WITH_SOABI)

target_link_libraries(_trace_object_boundaries_cython PRIVATE Python::Module Python::NumPy)

install(TARGETS _trace_object_boundaries_cython DESTINATION histomicstk/segmentation/label)
