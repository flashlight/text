cmake_minimum_required(VERSION 3.18)

if (NOT BUILD_SHARED_LIBS)
  message(FATAL_ERROR "Cannot build Python bindings with a static lib build: "
    "set BUILD_SHARED_LIBS to ON.")
endif()

# Since the Python libs and standalone Flashlight Text libs are built/installed
# to the same directory, set rpaths on the Python targets to be the current dir
if(APPLE)
  # macOS
  set(CMAKE_MACOSX_RPATH ON)
  set(_portable_rpath_origin "@loader_path")
else()
  # Linux
  set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
  set(_portable_rpath_origin $ORIGIN)
endif(APPLE)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG)
if (NOT pybind11_FOUND)
  message(STATUS "Could not find pybind11 -- downloading from source.")
  include(${PROJECT_SOURCE_DIR}/cmake/Buildpybind11.cmake)
endif()

function(add_pybind11_extension ext_name)
  string(REPLACE "_" ";" modlist ${ext_name})
  list(GET modlist -1 modname)
  list(REMOVE_AT modlist -1)
  if(modlist)
    string(REPLACE ";" "/" relpath "${modlist}")
  else()
    set(relpath "")
  endif()

  pybind11_add_module(
    ${ext_name}
    ${CMAKE_CURRENT_LIST_DIR}/${relpath}/_${modname}.cpp
    )

  target_link_libraries(
    ${ext_name}
    PUBLIC
    flashlight-text
    )

  target_include_directories(
    ${ext_name}
    PRIVATE
    ${PROJECT_SOURCE_DIR}
    )

  # Package build versus build bindings directly with CMake
  if (FL_TEXT_BUILD_PYTHON_PACKAGE)
    set_target_properties(${ext_name} PROPERTIES
      OUTPUT_NAME ${ext_name}
      BUILD_RPATH ${_portable_rpath_origin})
  else()
    if (CMAKE_LIBRARY_OUTPUT_DIRECTORY)
      set_target_properties(${ext_name} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${relpath})
    endif()
  endif()
endfunction()

add_pybind11_extension(flashlight_lib_text_decoder)
add_pybind11_extension(flashlight_lib_text_dictionary)

if (FL_TEXT_USE_KENLM)
  add_pybind11_extension(flashlight_lib_text_decoder_kenlm)
  target_link_libraries(
    flashlight_lib_text_decoder_kenlm
    PUBLIC
    flashlight-text-kenlm
    )
endif()
