/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "flashlight/lib/text/decoder/LexiconDecoder.h"
#include "flashlight/lib/text/decoder/LexiconFreeDecoder.h"
#include "flashlight/lib/text/decoder/LexiconFreeSeq2SeqDecoder.h"
#include "flashlight/lib/text/decoder/LexiconSeq2SeqDecoder.h"
#include "flashlight/lib/text/decoder/Utils.h"
#include "flashlight/lib/text/decoder/lm/ZeroLM.h"

namespace py = pybind11;
using namespace fl::lib::text;
using namespace py::literals;

PYBIND11_MAKE_OPAQUE(EmittingModelStatePtr);
PYBIND11_MAKE_OPAQUE(std::vector<EmittingModelStatePtr>);

namespace {

/**
 * A pybind11 "alias type" for abstract class LM, allowing one to subclass LM
 * with a custom LM defined purely in Python. For those who don't want to build
 * with KenLM, or have their own custom LM implementation.
 * See: https://pybind11.readthedocs.io/en/stable/advanced/classes.html
 *
 * TODO: ensure this works. Last time Jeff tried this there were slicing issues,
 * see https://github.com/pybind/pybind11/issues/1546 for workarounds.
 * This is low-pri since we assume most people can just build with KenLM.
 */
class PyLM : public LM {
  using LM::LM;

  // needed for pybind11 or else it won't compile
  using LMOutput = std::pair<LMStatePtr, float>;

  LMStatePtr start(bool startWithNothing) override {
    PYBIND11_OVERLOAD_PURE(LMStatePtr, LM, start, startWithNothing);
  }

  LMOutput score(const LMStatePtr& state, const int usrTokenIdx) override {
    PYBIND11_OVERLOAD_PURE(LMOutput, LM, score, state, usrTokenIdx);
  }

  LMOutput finish(const LMStatePtr& state) override {
    PYBIND11_OVERLOAD_PURE(LMOutput, LM, finish, state);
  }
};

/**
 * Using custom python LMState derived from LMState is not working with
 * custom python LM (derived from PyLM) because we need to to custing of LMState
 * in score and finish functions to the derived class
 * (for example vie obj.__class__ = CustomPyLMSTate) which cause the error
 * "TypeError: __class__ assignment: 'CustomPyLMState' deallocator differs
 * from 'flashlight.text.decoder._decoder.LMState'"
 * details see in https://github.com/pybind/pybind11/issues/1640
 * To define custom LM you can introduce map inside LM which maps LMstate to
 * additional state info (shared pointers pointing to the same underlying object
 * will have the same id in python in functions score and finish)
 *
 * ```python
 * from flashlight.lib.text.decoder import LM
 * class MyPyLM(LM):
 *      mapping_states = dict() # store simple additional int for each state
 *
 *      def __init__(self):
 *          LM.__init__(self)
 *
 *       def start(self, start_with_nothing):
 *          state = LMState()
 *          self.mapping_states[state] = 0
 *          return state
 *
 *      def score(self, state, index):
 *          outstate = state.child(index)
 *          if outstate not in self.mapping_states:
 *              self.mapping_states[outstate] = self.mapping_states[state] + 1
 *          return (outstate, -numpy.random.random())
 *
 *      def finish(self, state):
 *          outstate = state.child(-1)
 *          if outstate not in self.mapping_states:
 *              self.mapping_states[outstate] = self.mapping_states[state] + 1
 *          return (outstate, -1)
 * ```
 */
void LexiconDecoder_decodeStep(
    LexiconDecoder& decoder,
    uintptr_t emissions,
    int T,
    int N) {
  decoder.decodeStep(reinterpret_cast<const float*>(emissions), T, N);
}

std::vector<DecodeResult> LexiconDecoder_decode(
    LexiconDecoder& decoder,
    uintptr_t emissions,
    int T,
    int N) {
  return decoder.decode(reinterpret_cast<const float*>(emissions), T, N);
}

void LexiconFreeDecoder_decodeStep(
    LexiconFreeDecoder& decoder,
    uintptr_t emissions,
    int T,
    int N) {
  decoder.decodeStep(reinterpret_cast<const float*>(emissions), T, N);
}

std::vector<DecodeResult> LexiconFreeDecoder_decode(
    LexiconFreeDecoder& decoder,
    uintptr_t emissions,
    int T,
    int N) {
  return decoder.decode(reinterpret_cast<const float*>(emissions), T, N);
}

void LexiconSeq2SeqDecoder_decodeStep(
    LexiconSeq2SeqDecoder& decoder,
    uintptr_t emissions,
    int T,
    int N) {
  decoder.decodeStep(reinterpret_cast<const float*>(emissions), T, N);
}

void LexiconFreeSeq2SeqDecoder_decodeStep(
    LexiconFreeSeq2SeqDecoder& decoder,
    uintptr_t emissions,
    int T,
    int N) {
  decoder.decodeStep(reinterpret_cast<const float*>(emissions), T, N);
}

/*
 * Create an EmittingModelStatePtr from an arbitrary Python object. Since
 * py::object refcounts Python and C++ usages (via its copy ctor), we can create
 * a shared pointer straight away without worrying about lifetime.
 */
EmittingModelStatePtr createEmittingModelState(py::object obj) {
  auto s = std::make_shared<py::object>(std::move(obj));
  return std::static_pointer_cast<void>(s);
}

/*
 * Recover a Python object from an EmittingModelStatePtr. Refcounting of
 * py::object means we can return a copy.
 */
py::object getObjFromEmittingModelState(EmittingModelStatePtr state) {
  // The only way to create this type from Python is via createModelStatePtr,
  // which is guaranteed to store a py::object; this is a safe static cast.
  auto val = std::static_pointer_cast<py::object>(state);
  return *val;
}

} // namespace

PYBIND11_MODULE(flashlight_lib_text_decoder, m) {
  py::enum_<SmearingMode>(m, "SmearingMode")
      .value("NONE", SmearingMode::NONE)
      .value("MAX", SmearingMode::MAX)
      .value("LOGADD", SmearingMode::LOGADD);

  py::class_<TrieNode, TrieNodePtr>(m, "TrieNode")
      .def(py::init<int>(), "idx"_a)
      .def_readwrite("children", &TrieNode::children)
      .def_readwrite("idx", &TrieNode::idx)
      .def_readwrite("labels", &TrieNode::labels)
      .def_readwrite("scores", &TrieNode::scores)
      .def_readwrite("max_score", &TrieNode::maxScore);

  py::class_<Trie, TriePtr>(m, "Trie")
      .def(py::init<int, int>(), "max_children"_a, "root_idx"_a)
      .def("get_root", &Trie::getRoot)
      .def("insert", &Trie::insert, "indices"_a, "label"_a, "score"_a)
      .def("search", &Trie::search, "indices"_a)
      .def("smear", &Trie::smear, "smear_mode"_a);

  py::class_<LM, LMPtr, PyLM>(m, "LM")
      .def(py::init<>())
      .def("start", &LM::start, "start_with_nothing"_a)
      .def("score", &LM::score, "state"_a, "usr_token_idx"_a)
      .def("finish", &LM::finish, "state"_a);

  py::class_<LMState, LMStatePtr>(m, "LMState")
      .def(py::init<>())
      .def_readwrite("children", &LMState::children)
      .def("compare", &LMState::compare, "state"_a)
      .def("child", &LMState::child<LMState>, "usr_index"_a);

  py::class_<ZeroLM, ZeroLMPtr, LM>(m, "ZeroLM").def(py::init<>());

  py::enum_<CriterionType>(m, "CriterionType")
      .value("ASG", CriterionType::ASG)
      .value("CTC", CriterionType::CTC)
      .value("S2S", CriterionType::S2S);

  py::class_<LexiconDecoderOptions>(m, "LexiconDecoderOptions")
      .def(
          py::init<
              const int,
              const int,
              const double,
              const double,
              const double,
              const double,
              const double,
              const bool,
              const CriterionType>(),
          "beam_size"_a,
          "beam_size_token"_a,
          "beam_threshold"_a,
          "lm_weight"_a,
          "word_score"_a,
          "unk_score"_a,
          "sil_score"_a,
          "log_add"_a,
          "criterion_type"_a)
      .def_readwrite("beam_size", &LexiconDecoderOptions::beamSize)
      .def_readwrite("beam_size_token", &LexiconDecoderOptions::beamSizeToken)
      .def_readwrite("beam_threshold", &LexiconDecoderOptions::beamThreshold)
      .def_readwrite("lm_weight", &LexiconDecoderOptions::lmWeight)
      .def_readwrite("word_score", &LexiconDecoderOptions::wordScore)
      .def_readwrite("unk_score", &LexiconDecoderOptions::unkScore)
      .def_readwrite("sil_score", &LexiconDecoderOptions::silScore)
      .def_readwrite("log_add", &LexiconDecoderOptions::logAdd)
      .def_readwrite("criterion_type", &LexiconDecoderOptions::criterionType)
      .def(py::pickle(
          [](const LexiconDecoderOptions& p) { // __getstate__
            return py::make_tuple(
                p.beamSize,
                p.beamSizeToken,
                p.beamThreshold,
                p.lmWeight,
                p.wordScore,
                p.unkScore,
                p.silScore,
                p.logAdd,
                p.criterionType);
          },
          [](py::tuple t) { // __setstate__
            if (t.size() != 9) {
              throw std::runtime_error(
                  "Cannot run __setstate__ on LexiconDecoderOptions - "
                  "insufficient arguments provided.");
            }
            LexiconDecoderOptions opts = {
                t[0].cast<int>(), // beamSize
                t[1].cast<int>(), // beamSizeToken
                t[2].cast<double>(), // beamThreshold
                t[3].cast<double>(), // lmWeight
                t[4].cast<double>(), // wordScore
                t[5].cast<double>(), // unkScore
                t[6].cast<double>(), // silScore
                t[7].cast<bool>(), // logAdd
                t[8].cast<CriterionType>() // criterionType
            };
            return opts;
          }));

  py::class_<LexiconFreeDecoderOptions>(m, "LexiconFreeDecoderOptions")
      .def(
          py::init<
              const int,
              const int,
              const double,
              const double,
              const double,
              const bool,
              const CriterionType>(),
          "beam_size"_a,
          "beam_size_token"_a,
          "beam_threshold"_a,
          "lm_weight"_a,
          "sil_score"_a,
          "log_add"_a,
          "criterion_type"_a)
      .def_readwrite("beam_size", &LexiconFreeDecoderOptions::beamSize)
      .def_readwrite(
          "beam_size_token", &LexiconFreeDecoderOptions::beamSizeToken)
      .def_readwrite(
          "beam_threshold", &LexiconFreeDecoderOptions::beamThreshold)
      .def_readwrite("lm_weight", &LexiconFreeDecoderOptions::lmWeight)
      .def_readwrite("sil_score", &LexiconFreeDecoderOptions::silScore)
      .def_readwrite("log_add", &LexiconFreeDecoderOptions::logAdd)
      .def_readwrite(
          "criterion_type", &LexiconFreeDecoderOptions::criterionType)
      .def(py::pickle(
          [](const LexiconFreeDecoderOptions& p) { // __getstate__
            return py::make_tuple(
                p.beamSize,
                p.beamSizeToken,
                p.beamThreshold,
                p.lmWeight,
                p.silScore,
                p.logAdd,
                p.criterionType);
          },
          [](py::tuple t) { // __setstate__
            if (t.size() != 7) {
              throw std::runtime_error(
                  "Cannot run __setstate__ on LexiconFreeDecoderOptions - "
                  "insufficient arguments provided.");
            }
            LexiconFreeDecoderOptions opts = {
                t[0].cast<int>(), // beamSize
                t[1].cast<int>(), // beamSizeToken
                t[2].cast<double>(), // beamThreshold
                t[3].cast<double>(), // lmWeight
                t[4].cast<double>(), // silScore
                t[5].cast<bool>(), // logAdd
                t[6].cast<CriterionType>() // criterionType
            };
            return opts;
          }));

  py::class_<DecodeResult>(m, "DecodeResult")
      .def(py::init<int>(), "length"_a)
      .def_readwrite("score", &DecodeResult::score)
      .def_readwrite("emittingModelScore", &DecodeResult::emittingModelScore)
      .def_readwrite("lmScore", &DecodeResult::lmScore)
      .def_readwrite("words", &DecodeResult::words)
      .def_readwrite("tokens", &DecodeResult::tokens);

  // NB: `decode` and `decodeStep` expect raw emissions pointers.
  py::class_<LexiconDecoder>(m, "LexiconDecoder")
      .def(
          py::init<
              LexiconDecoderOptions,
              const TriePtr,
              const LMPtr,
              const int,
              const int,
              const int,
              const std::vector<float>&,
              const bool>(),
          "options"_a,
          "trie"_a,
          "lm"_a,
          "sil_token_idx"_a,
          "blank_token_idx"_a,
          "unk_token_idx"_a,
          "transitions"_a,
          "is_token_lm"_a)
      .def("decode_begin", &LexiconDecoder::decodeBegin)
      .def(
          "decode_step",
          &LexiconDecoder_decodeStep,
          "emissions"_a,
          "T"_a,
          "N"_a)
      .def("decode_end", &LexiconDecoder::decodeEnd)
      .def("decode", &LexiconDecoder_decode, "emissions"_a, "T"_a, "N"_a)
      .def("prune", &LexiconDecoder::prune, "look_back"_a = 0)
      .def(
          "get_best_hypothesis",
          &LexiconDecoder::getBestHypothesis,
          "look_back"_a = 0)
      .def("get_all_final_hypothesis", &LexiconDecoder::getAllFinalHypothesis);

  py::class_<LexiconFreeDecoder>(m, "LexiconFreeDecoder")
      .def(
          py::init<
              LexiconFreeDecoderOptions,
              const LMPtr,
              const int,
              const int,
              const std::vector<float>&>(),
          "options"_a,
          "lm"_a,
          "sil_token_idx"_a,
          "blank_token_idx"_a,
          "transitions"_a)
      .def("decode_begin", &LexiconFreeDecoder::decodeBegin)
      .def(
          "decode_step",
          &LexiconFreeDecoder_decodeStep,
          "emissions"_a,
          "T"_a,
          "N"_a)
      .def("decode_end", &LexiconFreeDecoder::decodeEnd)
      .def("decode", &LexiconFreeDecoder_decode, "emissions"_a, "T"_a, "N"_a)
      .def("prune", &LexiconFreeDecoder::prune, "look_back"_a = 0)
      .def(
          "get_best_hypothesis",
          &LexiconFreeDecoder::getBestHypothesis,
          "look_back"_a = 0)
      .def(
          "get_all_final_hypothesis",
          &LexiconFreeDecoder::getAllFinalHypothesis)
      .def("get_options", &LexiconFreeDecoder::getOptions)
      .def("get_sil_idx", &LexiconFreeDecoder::getSilIdx)
      .def("get_blank_idx", &LexiconFreeDecoder::getBlankIdx)
      .def("get_transitions", &LexiconFreeDecoder::getTransitions)
      .def(
          "get_options",
          &LexiconFreeDecoder::getOptions)
      .def(py::pickle(
          [](const LexiconFreeDecoder& p) { // __getstate__
            if (p.getAllFinalHypothesis().size() != 0) {
              throw std::runtime_error(
                  "LexiconFreeDecoder: cannot pickle decoder that has state");
            }
            if (!std::dynamic_pointer_cast<ZeroLM>(p.getLMPtr())) {
              throw std::runtime_error(
                  "LexiconFreeDecoder: cannot pickle a decoder with an "
                  "integrated language model that is not ZeroLM");
            }
            return py::make_tuple(
                p.getOptions(),
                p.getSilIdx(),
                p.getBlankIdx(),
                p.getTransitions());
          },
          [](py::tuple t) { // __setstate__
            if (t.size() != 4) {
              throw std::runtime_error(
                  "Cannot run __setstate__ on LexiconFreeDecoder - "
                  "insufficient arguments provided.");
            }

            return LexiconFreeDecoder(
                t[0].cast<LexiconFreeDecoderOptions>(), // options
                std::make_shared<ZeroLM>(), // lm
                t[1].cast<int>(), // silIdx
                t[2].cast<int>(), // blankIdx
                t[3].cast<std::vector<float>>() // transitions
            );
          }));

  // Seq2seq Decoding
  py::class_<LexiconSeq2SeqDecoderOptions>(m, "LexiconSeq2SeqDecoderOptions")
      .def(
          py::init<
              const int,
              const int,
              const double,
              const double,
              const double,
              const double,
              const bool>(),
          "beam_size"_a,
          "beam_size_token"_a,
          "beam_threshold"_a,
          "lm_weight"_a,
          "word_score"_a,
          "eos_score"_a,
          "log_add"_a)
      .def_readwrite("beam_size", &LexiconSeq2SeqDecoderOptions::beamSize)
      .def_readwrite(
          "beam_size_token", &LexiconSeq2SeqDecoderOptions::beamSizeToken)
      .def_readwrite(
          "beam_threshold", &LexiconSeq2SeqDecoderOptions::beamThreshold)
      .def_readwrite("lm_weight", &LexiconSeq2SeqDecoderOptions::lmWeight)
      .def_readwrite("word_score", &LexiconSeq2SeqDecoderOptions::wordScore)
      .def_readwrite("eos_score", &LexiconSeq2SeqDecoderOptions::eosScore)
      .def_readwrite("log_add", &LexiconSeq2SeqDecoderOptions::logAdd);

  py::class_<LexiconFreeSeq2SeqDecoderOptions>(
      m, "LexiconFreeSeq2SeqDecoderOptions")
      .def(
          py::init<
              const int,
              const int,
              const double,
              const double,
              const double,
              const bool>(),
          "beam_size"_a,
          "beam_size_token"_a,
          "beam_threshold"_a,
          "lm_weight"_a,
          "eos_score"_a,
          "log_add"_a)
      .def_readwrite("beam_size", &LexiconFreeSeq2SeqDecoderOptions::beamSize)
      .def_readwrite(
          "beam_size_token", &LexiconFreeSeq2SeqDecoderOptions::beamSizeToken)
      .def_readwrite(
          "beam_threshold", &LexiconFreeSeq2SeqDecoderOptions::beamThreshold)
      .def_readwrite("lm_weight", &LexiconFreeSeq2SeqDecoderOptions::lmWeight)
      .def_readwrite("eos_score", &LexiconFreeSeq2SeqDecoderOptions::eosScore)
      .def_readwrite("log_add", &LexiconFreeSeq2SeqDecoderOptions::logAdd);

  py::class_<LexiconSeq2SeqDecoder>(m, "LexiconSeq2SeqDecoder")
      .def(
          py::init<
              LexiconSeq2SeqDecoderOptions,
              const TriePtr,
              const LMPtr,
              const int,
              EmittingModelUpdateFunc,
              const int,
              const bool>(),
          "options"_a,
          "lm"_a,
          "trie"_a,
          "eos_idx"_a,
          "update_func"_a,
          "max_output_length"_a,
          "is_token_lm"_a)
      .def(
          "decode_step",
          &LexiconSeq2SeqDecoder_decodeStep,
          "emissions"_a,
          "T"_a,
          "N"_a)
      .def("prune", &LexiconSeq2SeqDecoder::prune, "look_back"_a = 0)
      .def(
          "get_best_hypothesis",
          &LexiconSeq2SeqDecoder::getBestHypothesis,
          "look_back"_a = 0)
      .def(
          "get_all_final_hypothesis",
          &LexiconSeq2SeqDecoder::getAllFinalHypothesis);

  // Constructor intentionally omitted -- not to be constructed directly.
  // create_model_state and get_model_state_obj should be used to create
  // instances of this type.
  py::class_<EmittingModelStatePtr>(m, "EmittingModelState");
  m.def("create_emitting_model_state", &createEmittingModelState);
  m.def("get_obj_from_emitting_model_state", &getObjFromEmittingModelState);

  py::bind_vector<std::vector<EmittingModelStatePtr>>(
      m, "VectorEmittingModelState");
  py::implicitly_convertible<py::list, std::vector<EmittingModelStatePtr>>();

  py::class_<LexiconFreeSeq2SeqDecoder>(m, "LexiconFreeSeq2SeqDecoder")
      .def(
          py::init<
              LexiconFreeSeq2SeqDecoderOptions,
              const LMPtr,
              const int,
              EmittingModelUpdateFunc,
              const int>(),
          "options"_a,
          "lm"_a,
          "eos_idx"_a,
          "update_func"_a,
          "max_output_length"_a)
      .def(
          "decode_step",
          &LexiconFreeSeq2SeqDecoder_decodeStep,
          "emissions"_a,
          "T"_a,
          "N"_a)
      .def("prune", &LexiconFreeSeq2SeqDecoder::prune, "look_back"_a = 0)
      .def(
          "get_best_hypothesis",
          &LexiconFreeSeq2SeqDecoder::getBestHypothesis,
          "look_back"_a = 0)
      .def(
          "get_all_final_hypothesis",
          &LexiconFreeSeq2SeqDecoder::getAllFinalHypothesis);
}
