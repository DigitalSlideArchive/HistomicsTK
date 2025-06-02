cmake_minimum_required(VERSION 3.15...3.26)

project(histomicstk LANGUAGES CXX)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module NumPy)

add_subdirectory(histomicstk/features)
add_subdirectory(histomicstk/segmentation/label)
add_subdirectory(histomicstk/segmentation/nuclear)
