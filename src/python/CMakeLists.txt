find_package(Python COMPONENTS Interpreter Development REQUIRED)

#if (APPLE)
#    find_package(pybind11 CONFIG REQUIRED)
#    add_library(spice MODULE ext_photonmapping.cpp)
#    target_link_libraries(spice PUBLIC pybind11::module pybind11::opt_size pybind11::thin_lto pybind11::windows_extras
#        tinyobjloader embree OpenMP::OpenMP_CXX pm)
#    pybind11_strip(spice)

#else()
# Detect the installed nanobind package and import it into CMake
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(libspice LTO ext_photonmapping_nano.cpp)
target_link_libraries(libspice PUBLIC tinyobjloader tinygltf embree OpenMP::OpenMP_CXX pm)
#endif()
