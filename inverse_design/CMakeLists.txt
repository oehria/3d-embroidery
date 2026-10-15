set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
cmake_minimum_required(VERSION 3.16)


add_executable(inv_design main.cpp)

set(DIRECTIONAL_ROOT "${CMAKE_SOURCE_DIR}/../external/Directional")
set(DIRECTIONAL_SOURCE_DIR "${DIRECTIONAL_ROOT}/include")
set(DIRECTIONAL_EXTERNAL "${DIRECTIONAL_ROOT}/external")

message(STATUS "DIRECTIONAL_ROOT: ${DIRECTIONAL_ROOT}")

include_directories(${DIRECTIONAL_SOURCE_DIR})
include_directories(${DIRECTIONAL_EXTERNAL}/eigen)

# libigl (exact submodule version, no find_package)
set(LIBIGL_DIR "${CMAKE_SOURCE_DIR}/../external/libigl")
add_subdirectory(${LIBIGL_DIR} libigl)


# Polyscope
add_subdirectory(${DIRECTIONAL_EXTERNAL}/polyscope "polyscope")

target_include_directories(inv_design PUBLIC
    ${CMAKE_SOURCE_DIR}/Directional
)

set(GEO_DIR "${CMAKE_SOURCE_DIR}/../external/geometry-central")
add_subdirectory(${GEO_DIR} geometry-central)

target_link_libraries(inv_design PUBLIC igl::core polyscope geometry-central)

