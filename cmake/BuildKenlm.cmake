cmake_minimum_required(VERSION 3.16)

include(FetchContent)

# TODO: swap to main repo once https://github.com/kpu/kenlm/pull/418 is merged
set(kenlm_URL https://github.com/jacobkahn/kenlm.git)
set(kenlm_TAG 5bf7b46558e1c5595bf3b8c9b0b1f9d8d257040a)

FetchContent_Declare(
    kenlm
    GIT_REPOSITORY ${kenlm_URL}
    GIT_TAG ${kenlm_TAG}
)

set(_BUILD_TESTING ${BUILD_TESTING}) # save if already set
# KenLM build options
set(ENABLE_INTERPOLATE OFF CACHE BOOL "" FORCE)
set(COMPILE_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(BUILD_PYTHON OFF CACHE BOOL "" FORCE)
set(FORCE_STATIC OFF CACHE BOOL "" FORCE)
set(BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(kenlm)

# includes are relative to the source dir
target_include_directories(kenlm PUBLIC $<BUILD_INTERFACE:${kenlm_SOURCE_DIR}>)
add_library(kenlm::kenlm ALIAS kenlm)

set(BUILD_TESTING ${_BUILD_TESTING}) # restore
