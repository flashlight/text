cmake_minimum_required(VERSION 3.16)

project(flashlight-text LANGUAGES CXX C VERSION 0.0.4)

include(CTest)

# ----------------------------- Setup -----------------------------
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(FL_TEXT_CODE_COVERAGE "Enable coverage reporting" OFF)

# Default directories for installation
set(FL_INSTALL_INC_DIR "include" CACHE PATH "Install path for headers")
set(FL_INSTALL_LIB_DIR "lib" CACHE PATH "Install path for libraries")
set(FL_INSTALL_BIN_DIR "bin" CACHE PATH "Install path for binaries")
# Other assets
set(FL_INSTALL_ASSETS_BASE_DIR "share/${PROJECT_NAME}")
set(FL_INSTALL_CMAKE_DIR "${FL_INSTALL_ASSETS_BASE_DIR}/cmake" CACHE PATH "Install path for CMake files")
set(FL_INSTALL_EXAMPLES_DIR "${FL_INSTALL_ASSETS_BASE_DIR}/examples" CACHE PATH "Install path for example files")
set(FL_INSTALL_DOC_DIR "${FL_INSTALL_ASSETS_BASE_DIR}/doc" CACHE PATH "Install path for documentation")

include(${PROJECT_SOURCE_DIR}/cmake/InternalUtils.cmake)

# ----------------------------- Configuration -----------------------------

option(FL_TEXT_BUILD_TESTS "Build tests" ON)
option(FL_TEXT_BUILD_STANDALONE "Build standalone installation" ON)
option(FL_TEXT_USE_KENLM "Use KenLM in flashlight libraries build" ON)
option(FL_TEXT_BUILD_PYTHON "Build Python bindings" OFF)
option(FL_TEXT_BUILD_PYTHON_PACKAGE "Build packaging-ready Python artifacts" OFF)

# ------------------------ Build ------------------------

add_library(flashlight-text)

target_include_directories(
  flashlight-text
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
)

set_target_properties(
  flashlight-text PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  )

target_compile_definitions(
  flashlight-text
  PUBLIC
  FL_TEXT_USE_KENLM=$<BOOL:${FL_TEXT_USE_KENLM}>
  FL_TEXT_DLL
)


if (FL_TEXT_USE_KENLM)
  # KenLM support in Flashlight Text is built as a separate standalone library.
  # This is to enable dynamically enabling KenLM support at runtime if
  # libkenlm is present and due to challenges packaging KenLM.
  add_library(flashlight-text-kenlm)
  add_library(flashlight::flashlight-text-kenlm ALIAS flashlight-text-kenlm)
endif()

# Project source
include(${PROJECT_SOURCE_DIR}/flashlight/lib/text/CMakeLists.txt)

if (FL_TEXT_BUILD_PYTHON)
  include(${PROJECT_SOURCE_DIR}/bindings/python/CMakeLists.txt)
endif()

add_library(flashlight::flashlight-text ALIAS flashlight-text)

# ------------------------ Tests + Code Coverage------------------------

if (FL_TEXT_BUILD_TESTS)
  enable_testing()
  include(${PROJECT_SOURCE_DIR}/cmake/TestUtils.cmake)
  add_subdirectory(${PROJECT_SOURCE_DIR}/flashlight/lib/text/test)
endif ()

if (FL_TEXT_CODE_COVERAGE)
  fl_text_add_coverage_to_target(TARGET flashlight-text)
endif()

# ------------------------ Installation ------------------------
set(_excluded_headers)             # header patterns to exclude
set(_install_libs flashlight-text) # lib targets to install

# Handle KenLM assets
if (NOT FL_TEXT_USE_KENLM)
  list(APPEND _excluded_headers "KenLM.*")
else()
  if (FL_TEXT_CODE_COVERAGE)
    fl_text_add_coverage_to_target(TARGET flashlight-text-kenlm)
  endif()
  list(APPEND _install_libs flashlight-text-kenlm)
endif()

# Install headers
install(
  DIRECTORY ${PROJECT_SOURCE_DIR}/flashlight/lib
  COMPONENT headers
  DESTINATION ${FL_INSTALL_INC_DIR}/flashlight
  FILES_MATCHING # preserve directory structure
  PATTERN  "*.h"
  PATTERN  "*.hpp"
  PATTERN "test*" EXCLUDE
  PATTERN ".git" EXCLUDE
  PATTERN "${_excluded_headers}" EXCLUDE
)

# Install libraries and create CMake config/targets files
fl_text_setup_install_targets(INSTALL_TARGETS ${_install_libs})
