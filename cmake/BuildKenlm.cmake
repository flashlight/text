cmake_minimum_required(VERSION 3.10.0)

# Required for KenLM to read ARPA files in compressed format
find_package(LibLZMA REQUIRED)
find_package(BZip2 REQUIRED)
find_package(ZLIB REQUIRED)

set(COMPRESSION_LIBS
  ${LIBLZMA_LIBRARIES}
  ${BZIP2_LIBRARIES}
  ${ZLIB_LIBRARIES}
  )

include(ExternalProject)

set(kenlm_TEMP_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/extern/kenlm)
set(kenlm_URL https://github.com/kpu/kenlm.git)
set(kenlm_BUILD ${CMAKE_CURRENT_BINARY_DIR}/third-party/kenlm)
set(kenlm_TAG 5cea457db26950a73d638425c183b368c06ed7c6)
set(kenlm_BINARY_DIR ${kenlm_BUILD}/src/kenlm-build)

if (BUILD_SHARED_LIBS)
  set(LIB_TYPE SHARED)
else()
  set(LIB_TYPE STATIC)
endif()

set(KENLM_LIB_PATH ${kenlm_BINARY_DIR}/lib/${CMAKE_${LIB_TYPE}_LIBRARY_PREFIX}kenlm${CMAKE_${LIB_TYPE}_LIBRARY_SUFFIX})
set(KENLM_UTIL_LIB_PATH ${kenlm_BINARY_DIR}/lib/${CMAKE_${LIB_TYPE}_LIBRARY_PREFIX}kenlm_util${CMAKE_${LIB_TYPE}_LIBRARY_SUFFIX})
set(KENLM_BUILT_LIBRARIES ${KENLM_LIB_PATH} ${KENLM_UTIL_LIB_PATH})

if (NOT TARGET kenlm)
  # Download kenlm
  ExternalProject_Add(
    kenlm
    PREFIX ${kenlm_BUILD}
    GIT_REPOSITORY ${kenlm_URL}
    GIT_TAG ${kenlm_TAG}
    BUILD_BYPRODUCTS
      ${KENLM_BUILT_LIBRARIES}
    CMAKE_CACHE_ARGS
      -DBUILD_SHARED_LIBS:BOOL=${BUILD_SHARED_LIBS}
      -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
      -DCMAKE_INSTALL_PREFIX:PATH=${kenlm_TEMP_INSTALL_DIR}
      )
endif ()

# Install the install executed at build time
install(DIRECTORY ${kenlm_TEMP_INSTALL_DIR}/include DESTINATION ${CMAKE_INSTALL_PREFIX})
install(DIRECTORY ${kenlm_TEMP_INSTALL_DIR}/lib DESTINATION ${CMAKE_INSTALL_PREFIX})
install(DIRECTORY ${kenlm_TEMP_INSTALL_DIR}/bin DESTINATION ${CMAKE_INSTALL_PREFIX})

set(KENLM_INCLUDE_DIRS "${kenlm_TEMP_INSTALL_DIR}/include;${kenlm_TEMP_INSTALL_DIR}/include/kenlm")
file(MAKE_DIRECTORY ${KENLM_INCLUDE_DIRS})

set(KENLM_LIBRARIES ${KENLM_BUILT_LIBRARIES} ${COMPRESSION_LIBS})

if (NOT TARGET kenlm::kenlm)
  add_library(kenlm::kenlm ${LIB_TYPE} IMPORTED)
  set_property(TARGET kenlm::kenlm PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${KENLM_INCLUDE_DIRS})
  set_property(TARGET kenlm::kenlm PROPERTY IMPORTED_LOCATION ${KENLM_LIB_PATH})
  add_dependencies(kenlm::kenlm kenlm)
endif()

if (NOT TARGET kenlm::kenlm_util)
  add_library(kenlm::kenlm_util ${LIB_TYPE} IMPORTED)
  set_property(TARGET kenlm::kenlm_util PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${KENLM_INCLUDE_DIRS})
  set_property(TARGET kenlm::kenlm_util PROPERTY IMPORTED_LOCATION ${KENLM_UTIL_LIB_PATH})
  set_property(TARGET kenlm::kenlm_util PROPERTY IMPORTED_LINK_INTERFACE_LIBRARIES ${COMPRESSION_LIBS})
  add_dependencies(kenlm::kenlm_util kenlm)
endif()
