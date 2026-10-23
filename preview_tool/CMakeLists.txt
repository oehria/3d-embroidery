cmake_minimum_required(VERSION 3.5...3.27)

# Set policies to be compatible with older CMake versions
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

include(FetchContent)

project(preview_tool)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# libigl
FetchContent_Declare(
  libigl
  GIT_REPOSITORY https://github.com/libigl/libigl.git
  GIT_TAG v2.4.0)
# polyscope
FetchContent_Declare(
  polyscope
  GIT_REPOSITORY https://github.com/nmwsharp/polyscope.git
  GIT_TAG v2.3.0)

# Eigen3
find_package(Eigen3 NO_MODULE)

FetchContent_MakeAvailable(libigl polyscope)

file(
  GLOB
  SRC_FILES
  *.cpp
  geom/*.cpp
  utils/*.cpp)
add_executable(${PROJECT_NAME} ${SRC_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC polyscope igl::core)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR})

target_link_options(${PROJECT_NAME} PRIVATE -L /opt/homebrew/opt/llvm/lib/c++/)
target_link_directories(${PROJECT_NAME} PUBLIC /opt/homebrew/opt/libomp/lib)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -gdwarf-4")
endif()