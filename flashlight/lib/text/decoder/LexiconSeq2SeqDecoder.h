/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <optional>
#include <unordered_map>

#include "flashlight/lib/text/Defines.h"
#include "flashlight/lib/text/decoder/Decoder.h"
#include "flashlight/lib/text/decoder/Trie.h"
#include "flashlight/lib/text/decoder/Utils.h"
#include "flashlight/lib/text/decoder/lm/LM.h"

namespace fl {
namespace lib {
namespace text {

struct LexiconSeq2SeqDecoderOptions {
  int beamSize; // Maximum number of hypothesis we hold after each step
  int beamSizeToken; // Maximum number of tokens we consider at each step
  double beamThreshold; // Threshold to prune hypothesis
  double lmWeight; // Weight of lm
  double wordScore; // Word insertion score
  double eosScore; // Score for inserting an EOS
  bool logAdd; // If or not use logadd when merging hypothesis
};

/**
 * LexiconSeq2SeqDecoderState stores information for each hypothesis in the
 * beam.
 */
struct LexiconSeq2SeqDecoderState {
  double score; // Accumulated total score so far
  LMStatePtr lmState; // Language model state
  const TrieNode* lex;
  const LexiconSeq2SeqDecoderState* parent; // Parent hypothesis
  int token; // Label of token
  int word;
  EmittingModelStatePtr emittingModelState; // Acoustic model state

  double emittingModelScore; // Accumulated AM score so far
  double lmScore; // Accumulated LM score so far
  std::optional<int> prevHypIdx; // Index of the hyp in the beam that lead here

  LexiconSeq2SeqDecoderState(
      const double score,
      const LMStatePtr& lmState,
      const TrieNode* lex,
      const LexiconSeq2SeqDecoderState* parent,
      const int token,
      const int word,
      const EmittingModelStatePtr& emittingModelState,
      const double emittingModelScore = 0,
      const double lmScore = 0,
      const std::optional<int> prevHypIdx = std::nullopt)
      : score(score),
        lmState(lmState),
        lex(lex),
        parent(parent),
        token(token),
        word(word),
        emittingModelState(emittingModelState),
        emittingModelScore(emittingModelScore),
        lmScore(lmScore),
        prevHypIdx(prevHypIdx) {}

  LexiconSeq2SeqDecoderState()
      : score(0),
        lmState(nullptr),
        lex(nullptr),
        parent(nullptr),
        token(-1),
        word(-1),
        emittingModelState(nullptr),
        emittingModelScore(0.),
        lmScore(0.),
        prevHypIdx(std::nullopt) {}

  int compareNoScoreStates(const LexiconSeq2SeqDecoderState* node) const {
    int lmCmp = lmState->compare(node->lmState);
    if (lmCmp != 0) {
      return lmCmp > 0 ? 1 : -1;
    } else if (lex != node->lex) {
      return lex > node->lex ? 1 : -1;
    } else if (token != node->token) {
      return token > node->token ? 1 : -1;
    }
    return 0;
  }

  int getWord() const {
    return word;
  }
};

/**
 * Decoder implements a beam seach decoder that finds the token transcription
 * W maximizing:
 *
 * AM(W) + lmWeight_ * log(P_{lm}(W)) + eosScore_ * |W_last == EOS|
 *
 * where P_{lm}(W) is the language model score. The transcription W is
 * constrained by a lexicon. The language model may operate at word-level
 * (isLmToken=false) or token-level (isLmToken=true).
 *
 * TODO: Doesn't support online decoding now.
 *
 */
class FL_TEXT_API LexiconSeq2SeqDecoder : public Decoder {
 public:
  LexiconSeq2SeqDecoder(
      LexiconSeq2SeqDecoderOptions opt,
      const TriePtr& lexicon,
      const LMPtr& lm,
      const int eos,
      EmittingModelUpdateFunc emittingModelUpdateFunc,
      const int maxOutputLength,
      const bool isLmToken)
      : opt_(std::move(opt)),
        lm_(lm),
        lexicon_(lexicon),
        eos_(eos),
        emittingModelUpdateFunc_(emittingModelUpdateFunc),
        maxOutputLength_(maxOutputLength),
        isLmToken_(isLmToken) {}

  void decodeStep(const float* emissions, int T, int N) override;

  void prune(int lookBack = 0) override;

  int nDecodedFramesInBuffer() const override;

  DecodeResult getBestHypothesis(int lookBack = 0) const override;

  std::vector<DecodeResult> getAllFinalHypothesis() const override;

 protected:
  LexiconSeq2SeqDecoderOptions opt_;
  LMPtr lm_;
  TriePtr lexicon_;
  int eos_;
  EmittingModelUpdateFunc emittingModelUpdateFunc_;
  // token indices for each hypo in the beam at the current timestep
  std::vector<int> rawY_;
  // the previous hypo index to which the current hypotheses is added
  std::vector<int> rawBeamIdx_;
  // states for each hypo in the beam at the current timestep
  std::vector<EmittingModelStatePtr> rawPrevStates_;
  int maxOutputLength_;
  bool isLmToken_;

  std::vector<LexiconSeq2SeqDecoderState> candidates_;
  std::vector<LexiconSeq2SeqDecoderState*> candidatePtrs_;
  double candidatesBestScore_;

  std::unordered_map<int, std::vector<LexiconSeq2SeqDecoderState>> hyp_;
};

} // namespace text
} // namespace lib
} // namespace fl
