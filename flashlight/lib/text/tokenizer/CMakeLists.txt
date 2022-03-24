cmake_minimum_required(VERSION 3.10)

target_sources(
  flashlight-text
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/PartialFileReader.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Tokenizer.cpp
  )

find_package(Threads REQUIRED)
target_link_libraries(flashlight-text PRIVATE Threads::Threads)
