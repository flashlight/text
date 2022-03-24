cmake_minimum_required(VERSION 3.10)

target_sources(
  flashlight-text
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/Dictionary.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Utils.cpp
  )
