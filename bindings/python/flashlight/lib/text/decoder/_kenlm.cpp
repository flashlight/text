/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "flashlight/lib/text/decoder/lm/KenLM.h"

#include <string>

namespace py = pybind11;
using namespace fl::lib::text;
using namespace py::literals;

PYBIND11_MODULE(flashlight_lib_text_decoder_kenlm, m) {
  py::class_<KenLM, KenLMPtr, LM>(m, "KenLM")
      .def(
          py::init<const std::string&, const Dictionary&>(),
          "path"_a,
          "usr_token_dict"_a);
}
