/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <unordered_map>

#include "flashlight/lib/text/decoder/Decoder.h"
#include "flashlight/lib/text/decoder/Utils.h"
#include "flashlight/lib/text/decoder/lm/LM.h"

namespace fl {
namespace lib {
namespace text {

struct LexiconFreeSeq2SeqDecoderOptions {
  int beamSize; // Maximum number of hypothesis we hold after each step
  int beamSizeToken; // Maximum number of tokens we consider at each step
  double beamThreshold; // Threshold to prune hypothesis
  double lmWeight; // Weight of lm
  double eosScore; // Score for inserting an EOS
  bool logAdd; // If or not use logadd when merging hypothesis
};

/**
 * LexiconFreeSeq2SeqDecoderState stores information for each hypothesis in the
 * beam.
 */
struct LexiconFreeSeq2SeqDecoderState {
  double score; // Accumulated total score so far
  LMStatePtr lmState; // Language model state
  const LexiconFreeSeq2SeqDecoderState* parent; // Parent hypothesis
  int token; // Label of token
  EmittingModelStatePtr emittingModelState; // Emitting model state

  double emittingModelScore; // Accumulated emitting model score so far
  double lmScore; // Accumulated LM score so far

  LexiconFreeSeq2SeqDecoderState(
      const double score,
      const LMStatePtr& lmState,
      const LexiconFreeSeq2SeqDecoderState* parent,
      const int token,
      const EmittingModelStatePtr& emittingModelState = nullptr,
      const double emittingModelScore = 0,
      const double lmScore = 0)
      : score(score),
        lmState(lmState),
        parent(parent),
        token(token),
        emittingModelState(emittingModelState),
        emittingModelScore(emittingModelScore),
        lmScore(lmScore) {}

  LexiconFreeSeq2SeqDecoderState()
      : score(0),
        lmState(nullptr),
        parent(nullptr),
        token(-1),
        emittingModelState(nullptr),
        emittingModelScore(0.),
        lmScore(0.) {}

  int compareNoScoreStates(const LexiconFreeSeq2SeqDecoderState* node) const {
    return lmState->compare(node->lmState);
  }

  int getWord() const {
    return -1;
  }
};

/**
 * Decoder implements a beam seach decoder that finds the token transcription
 * W maximizing:
 *
 * AM(W) + lmWeight_ * log(P_{lm}(W)) + eosScore_ * |W_last == EOS|
 *
 * where P_{lm}(W) is the language model score. The sequence of tokens is not
 * constrained by a lexicon, and thus the language model must operate at
 * token-level.
 *
 * TODO: Doesn't support online decoding now.
 *
 */
class LexiconFreeSeq2SeqDecoder : public Decoder {
 public:
  LexiconFreeSeq2SeqDecoder(
      LexiconFreeSeq2SeqDecoderOptions opt,
      const LMPtr& lm,
      const int eos,
      EmittingModelUpdateFunc emittingModelUpdateFunc,
      const int maxOutputLength)
      : opt_(std::move(opt)),
        lm_(lm),
        eos_(eos),
        emittingModelUpdateFunc_(emittingModelUpdateFunc),
        maxOutputLength_(maxOutputLength) {}

  void decodeStep(const float* emissions, int T, int N) override;

  void prune(int lookBack = 0) override;

  int nDecodedFramesInBuffer() const override;

  DecodeResult getBestHypothesis(int lookBack = 0) const override;

  std::vector<DecodeResult> getAllFinalHypothesis() const override;

 protected:
  LexiconFreeSeq2SeqDecoderOptions opt_;
  LMPtr lm_;
  int eos_;
  EmittingModelUpdateFunc emittingModelUpdateFunc_;
  std::vector<int> rawY_;
  std::vector<EmittingModelStatePtr> rawPrevStates_;
  int maxOutputLength_;

  std::vector<LexiconFreeSeq2SeqDecoderState> candidates_;
  std::vector<LexiconFreeSeq2SeqDecoderState*> candidatePtrs_;
  double candidatesBestScore_;

  std::unordered_map<int, std::vector<LexiconFreeSeq2SeqDecoderState>> hyp_;
};
} // namespace text
} // namespace lib
} // namespace fl
