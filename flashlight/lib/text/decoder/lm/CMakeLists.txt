cmake_minimum_required(VERSION 3.16)

set(FL_TEXT_KENLM_MAX_ORDER 6 CACHE STRING "Maximum ngram order for KenLM")

target_sources(
  flashlight-text
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/ConvLM.cpp
  ${CMAKE_CURRENT_LIST_DIR}/ZeroLM.cpp
  )

# ------------------------- KenLM-specific -------------------------

if (FL_TEXT_USE_KENLM)
  find_package(kenlm CONFIG)
  if (NOT TARGET kenlm::kenlm)
    if (FL_TEXT_BUILD_PYTHON)
      # ONLY for Python installation, try to use Findkenlm.cmake to locate
      # KenLM libs (and headers) since libkenlm's in site-packages w/o a config
      find_package(kenlm MODULE REQUIRED) # use Findkenlm.cmake
    else()
      # Not building Python bindings - download standalone or fail
      if (FL_TEXT_BUILD_STANDALONE)
        message(STATUS "KenLM not found - will download and build from source")
        include(${PROJECT_SOURCE_DIR}/cmake/BuildKenlm.cmake)
      else()
        message(WARNING "KenLM not found but FL_TEXT_USE_KENLM enabled. "
          "Install KenLM to continue.")
        find_package(kenlm CONFIG REQUIRED) # find again to display full errors
      endif() # FL_TEXT_BUILD_STANDALONE
    endif() # FL_TEXT_BUILD_PYTHON
  endif() # NOT TARGET kenlm::kenlm

  target_sources(
    flashlight-text-kenlm
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/KenLM.cpp
    )

  target_link_libraries(
    flashlight-text-kenlm
    PRIVATE
    flashlight-text
    kenlm::kenlm
    )

  target_compile_definitions(
    flashlight-text-kenlm
    PRIVATE
    KENLM_MAX_ORDER=${FL_TEXT_KENLM_MAX_ORDER}
    )
endif()
