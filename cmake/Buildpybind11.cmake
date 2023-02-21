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
