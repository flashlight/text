cmake_minimum_required(VERSION 3.10)

set(DIR ${CMAKE_CURRENT_LIST_DIR})
set(LIBS flashlight-text)

build_test(SRC ${DIR}/dictionary/DictionaryTest.cpp LIBS ${LIBS}
  PREPROC "DICTIONARY_TEST_DATADIR=\"${DIR}/dictionary\"")
build_test(SRC ${DIR}/tokenizer/TokenizerTest.cpp LIBS ${LIBS}
  PREPROC "TOKENIZER_TEST_DATADIR=\"${DIR}/tokenizer\"")
build_test(SRC ${DIR}/StringTest.cpp LIBS ${LIBS})
build_test(SRC ${DIR}/decoder/Seq2SeqDecoderTest.cpp LIBS ${LIBS})

if (FL_TEXT_USE_KENLM)
  build_test(
    SRC ${DIR}/decoder/DecoderTest.cpp
    LIBS ${LIBS} flashlight-text-kenlm
    PREPROC "DECODER_TEST_DATADIR=\"${DIR}/decoder/data\"")
endif()

if (FL_TEXT_BUILD_PYTHON)
  find_package(PythonInterp REQUIRED)
  add_test(
    NAME PythonBindingsTest
    COMMAND ${PYTHON_EXECUTABLE} -m unittest discover -v ${PROJECT_SOURCE_DIR}/bindings/python/test
  )
  set_tests_properties(PythonBindingsTest PROPERTIES
    ENVIRONMENT
    "DATA_DIR=${PROJECT_SOURCE_DIR}/flashlight/lib/text/test/decoder/data;USE_KENLM=$<BOOL:${FL_TEXT_USE_KENLM}>;"
  )
endif()
