cmake_minimum_required(VERSION 3.10)

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
    # Some installations only have a single libkenlm. When installing
    # Python bindings, no kenlm_utils lib is present
    set(KENLM_DISABLE_UTILS_SEARCH ${FL_TEXT_BUILD_PYTHON_PACKAGE})
    find_package(kenlm)
    if (NOT kenlm_FOUND)
      if (FL_TEXT_BUILD_STANDALONE)
        message(STATUS "KenLM not found - will download and build from source")
        include(${PROJECT_SOURCE_DIR}/cmake/BuildKenlm.cmake)
      else()
        message(FATAL_ERROR "KenLM not found but FL_TEXT_USE_KENLM enabled. "
          "Install KenLM or set the KENLM_ROOT environment variable.")
      endif()
    endif()
    if (FL_TEXT_BUILD_STANDALONE)
      # Install the module so downstream projects can find the lib
      install(
        FILES ${PROJECT_SOURCE_DIR}/cmake/Findkenlm.cmake
        DESTINATION ${FL_INSTALL_CMAKE_DIR}
        )
    endif()
  endif()

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
    $<$<NOT:$<BOOL:${FL_TEXT_BUILD_PYTHON_PACKAGE}>>:kenlm::kenlm_util>
    )

  target_compile_definitions(
    flashlight-text-kenlm
    PRIVATE
    KENLM_MAX_ORDER=${FL_TEXT_KENLM_MAX_ORDER}
    )
endif()
