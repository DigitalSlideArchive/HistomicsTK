cmake_minimum_required(VERSION 3.6.2)

project(histomicstk)

find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(PythonExtensions REQUIRED)

add_subdirectory(histomicstk/features)
add_subdirectory(histomicstk/segmentation/label)
add_subdirectory(histomicstk/segmentation/nuclear)
