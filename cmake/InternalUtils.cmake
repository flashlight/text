function(fl_text_add_coverage_to_target)
  set(oneValueArgs TARGET)
  cmake_parse_arguments(add_coverage_to_target "${options}" "${oneValueArgs}"
    "${multiValueArgs}" ${ARGN})

  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # Add required flags (GCC & LLVM/Clang)
    target_compile_options(${add_coverage_to_target_TARGET} PUBLIC
      -O0 # TODO: reconcile this with CMake modes for something cleaner
      -g
      $<$<COMPILE_LANGUAGE:CXX>:--coverage>
      )
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.13)
      target_link_options(${add_coverage_to_target_TARGET}
        PUBLIC
        $<$<COMPILE_LANGUAGE:CXX>:--coverage>)
    else()
      target_link_libraries(${add_coverage_to_target_TARGET}
        PUBLIC
        --coverage)
    endif()
  endif()
endfunction(fl_text_add_coverage_to_target)

function(fl_text_setup_install_targets)
  set(multiValueArgs INSTALL_TARGETS INSTALL_HEADERS)
  cmake_parse_arguments(setup_install_targets "${options}" "${oneValueArgs}"
    "${multiValueArgs}" ${ARGN})

  list(LENGTH setup_install_targets_INSTALL_TARGETS TARGETS_LENGTH)
  if (${TARGETS_LENGTH} EQUAL 0)
    message(FATAL_ERROR "setup_install_targets called with "
      "empty targets list.")
  endif()

  # Main target
  install(
    TARGETS ${setup_install_targets_INSTALL_TARGETS}
    EXPORT flashlight-text-targets
    COMPONENT flashlight-text
    PUBLIC_HEADER DESTINATION fl
    RUNTIME DESTINATION ${FL_INSTALL_BIN_DIR}
    LIBRARY DESTINATION ${FL_INSTALL_LIB_DIR}
    ARCHIVE DESTINATION ${FL_INSTALL_LIB_DIR}
    FRAMEWORK DESTINATION framework
    INCLUDES DESTINATION ${FL_INSTALL_INC_DIR}
    )

  # Write and install targets file
  install(
    EXPORT flashlight-text-targets
    NAMESPACE flashlight::
    DESTINATION ${FL_INSTALL_CMAKE_DIR}
    COMPONENT flashlight-text
    )

  # Write config file (used by projects including fl, such as examples)
  include(CMakePackageConfigHelpers)
  set(INCLUDE_DIRS include)
  set(CMAKE_DIR ${FL_INSTALL_CMAKE_DIR})
  configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/cmake/flashlight-text-config.cmake.in
    cmake/install/${FL_CONFIG_CMAKE_BUILD_DIR}/flashlight-text-config.cmake
    INSTALL_DESTINATION
    ${FL_INSTALL_CMAKE_DIR}
    PATH_VARS INCLUDE_DIRS CMAKE_DIR
    )
  write_basic_package_version_file(
    cmake/install/${FL_CONFIG_CMAKE_BUILD_DIR}/flashlight-text-config-version.cmake
    COMPATIBILITY SameMajorVersion
    )
  install(FILES
    ${PROJECT_BINARY_DIR}/cmake/install/flashlight-text-config.cmake
    ${PROJECT_BINARY_DIR}/cmake/install/flashlight-text-config-version.cmake
    DESTINATION ${FL_INSTALL_CMAKE_DIR}
    COMPONENT flashlight-text
    )
  set_target_properties(${setup_install_targets_INSTALL_TARGETS} PROPERTIES
    VERSION "${flashlight-text_VERSION}"
    SOVERSION "${flashlight-text_VERSION_MAJOR}")
endfunction(fl_text_setup_install_targets)
