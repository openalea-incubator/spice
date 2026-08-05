cmake_minimum_required(VERSION 3.20)
project(photon_mapping LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 26)

option(BUILD_TESTS "build c++ tests" OFF)
option(BUILD_CXX_EXAMPLE "build c++ example" OFF)

option(HIP "GPU support with HIP" OFF)

# --- Build setup
find_package(TBB CONFIG REQUIRED)
find_package(Threads REQUIRED)

find_package(OpenMP REQUIRED)
if(NOT OpenMP_CXX_FOUND)
    message(WARNING "failed to find OpenMP")
endif()

# apple newer C++ features with old SDK
if(APPLE)
    set(CMAKE_CXX_FLAGS "${CXXFLAGS} -D_LIBCPP_DISABLE_AVAILABILITY")
    set(CMAKE_OSX_DEPLOYMENT_TARGET "10.13")
endif()

# Embree4
find_package(embree 4.0 REQUIRED)
if (${embree_FOUND})
  message(STATUS "Found Embree")
  set("EMBREE_MAX_ISA" ON)
else()
  message(FATAL_ERROR "Could not find Embree")
endif()

# find tinyobjloader & tinygltf
include(FetchContent)

set(TINYOBJLOADER_BUILD_TEST_LOADER OFF CACHE INTERNAL "") 

FetchContent_Declare(
    tinyobjloader
    GIT_REPOSITORY https://github.com/tinyobjloader/tinyobjloader.git
    GIT_TAG        release 
)

set(TINYGLTF_BUILD_EXAMPLES OFF CACHE INTERNAL "") 

FetchContent_Declare(
    tinygltf
    GIT_REPOSITORY https://github.com/syoyo/tinygltf.git
    GIT_TAG        release 
)

# Bring both packages into the build tree
FetchContent_MakeAvailable(tinyobjloader tinygltf)

if(HIP)
    find_package(hiprt CONFIG REQUIRED)
endif()

add_subdirectory("src/cpp/include")

# pm(photon_mapping)
add_library(pm INTERFACE src/python/ext_photonmapping_nano.cpp)
# target_include_directories(pm INTERFACE "src/cpp/include")

set_target_properties(pm
        PROPERTIES
        CXX_STANDARD 26
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS YES
        )


target_link_libraries(pm INTERFACE TBB::tbb OpenMP::OpenMP_CXX embree Threads::Threads photon_map primitive triangle image)

# wrapper
add_subdirectory("src/python/")

# tests
if(BUILD_TESTS)
    add_subdirectory("src/cpp/test")
endif()

# examples
if(BUILD_CXX_EXAMPLE)
    add_subdirectory("examples")
endif ()

# building lib
find_package(Python COMPONENTS Interpreter Development REQUIRED)
# find_package(pybind11 CONFIG REQUIRED)
# Detect the installed nanobind package and import it into CMake
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

install(TARGETS libspice
        COMPONENT python
        LIBRARY DESTINATION "${SKBUILD_PLATLIB_DIR}/openalea/spice/")
