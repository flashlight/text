cmake_minimum_required(VERSION 3.16)

include(FetchContent)

set(pybind11_URL https://github.com/pybind/pybind11.git)
set(pybind11_TAG v2.10.3)

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY ${pybind11_URL}
    GIT_TAG ${pybind11_TAG}
)

FetchContent_MakeAvailable(pybind11)
add_subdirectory(${pybind11_SOURCE_DIR})

# # Download pybind11
# ExternalProject_Add(
#   pybind11
#   PREFIX pybind11
#   GIT_REPOSITORY ${pybind11_URL}
#   GIT_TAG ${pybind11_TAG}
#   BUILD_IN_SOURCE 0
#   CONFIGURE_COMMAND ""
#   BUILD_COMMAND ""
#   INSTALL_COMMAND ""
# )

# ExternalProject_Get_Property(pybind11 SOURCE_DIR)
# set(pybind11_INCLUDE_DIR "${SOURCE_DIR}/include")
