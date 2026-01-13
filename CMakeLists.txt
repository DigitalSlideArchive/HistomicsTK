cmake_minimum_required(VERSION 3.15...3.26)

project(histomicstk LANGUAGES CXX)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module NumPy)

add_subdirectory(histomicstk/features)
add_subdirectory(histomicstk/segmentation/label)
add_subdirectory(histomicstk/segmentation/nuclear)
# --- Build Rust extension ---
message(STATUS "Building Rust extension with Cargo")
set(ENV{PYTHON_SYS_EXECUTABLE} ${Python3_EXECUTABLE})
set(ENV{PYO3_PYTHON}         ${Python3_EXECUTABLE})

if(APPLE)
    # on macOS tell Rust to leave Python symbols undefined so they're
    # picked up at runtime by the Python interpreter
    set(ENV{RUSTFLAGS}          "-C link-arg=-undefined -C link-arg=dynamic_lookup")
endif()

execute_process(
    COMMAND cargo build --release
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    RESULT_VARIABLE cargo_result
)

# clean up
unset(ENV{PYTHON_SYS_EXECUTABLE})
unset(ENV{PYO3_PYTHON})
if(APPLE)
    unset(ENV{RUSTFLAGS})
endif()
if(NOT cargo_result EQUAL 0)
    message(FATAL_ERROR "Cargo build failed ${Python3_EXECUTABLE}")
endif()

# Find and rename the built Rust shared library for Python import
if(APPLE)
    set(RUST_LIB "${CMAKE_SOURCE_DIR}/target/release/lib_lib.dylib")
    set(PYTHON_LIB "${CMAKE_SOURCE_DIR}/histomicstk/_rust.so")
    file(COPY ${RUST_LIB} DESTINATION "${CMAKE_SOURCE_DIR}/histomicstk")
    file(RENAME "${CMAKE_SOURCE_DIR}/histomicstk/lib_lib.dylib" ${PYTHON_LIB})
elseif(UNIX)
    set(RUST_LIB "${CMAKE_SOURCE_DIR}/target/release/lib_lib.so")
    set(PYTHON_LIB "${CMAKE_SOURCE_DIR}/histomicstk/_rust.so")
    file(COPY ${RUST_LIB} DESTINATION "${CMAKE_SOURCE_DIR}/histomicstk")
    file(RENAME "${CMAKE_SOURCE_DIR}/histomicstk/lib_lib.so" ${PYTHON_LIB})
elseif(WIN32)
    set(RUST_LIB "${CMAKE_SOURCE_DIR}/target/release/_lib.dll")
    set(PYTHON_LIB "${CMAKE_SOURCE_DIR}/histomicstk/_rust.pyd")
    file(COPY ${RUST_LIB} DESTINATION ${PYTHON_LIB})
    file(RENAME "${CMAKE_SOURCE_DIR}/histomicstk/_lib.dll" ${PYTHON_LIB})
endif()

install(FILES ${PYTHON_LIB} DESTINATION histomicstk)