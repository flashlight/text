/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "flashlight/lib/text/decoder/lm/LM.h"

namespace fl {
namespace lib {
namespace text {

/* ===================== Definitions ===================== */

const double kNegativeInfinity = -std::numeric_limits<double>::infinity();
const int kLookBackLimit = 100;

struct DecodeResult {
  double score;
  double emittingModelScore;
  double lmScore;
  std::vector<int> words;
  std::vector<int> tokens;

  explicit DecodeResult(int length = 0)
      : score(0), words(length, -1), tokens(length, -1) {}
};

/**
 * An opaque type used to represent stateful components of an autoregressive
 * model that need to be maintained for incremental decoding.
 *
 * For example: consider an RNN, which has different hidden states depending on
 * the prior state. This hidden state could be a single tensor, a struct
 * containing tensors, or any arbitrary type.
 */
using EmittingModelStatePtr = std::shared_ptr<void>;

/**
 * A callback type used to represent a closure called for each step in
 * autoregressive decoding.
 *
 * For each timestep in autoregressive coding, each potential candidate for the
 * next timestep for each element in the beam needs to be scored. The size of
 * the set of "current" autoregressive state used to score those candidates is
 * equal to the number of candidates in the beam.
 *
 * Decoder state is intentionally not implemented to be thread-safe; for a given
 * decoder instance in a single thread, closures of this type will not
 * be invoked from multiple threads simultaneously.
 *
 * An example implementation of such a closure with optimizations for memory use
 * and efficiency might peform the following steps:
 * 1. Collect the emissions for decoder input (e.g. encoder output) at the first
 *    time step, and save those in a buffer accessible to other invocations of
 *    the update function.
 * 2. Forward the candidate tokens through the next autoregressive step. For
 *    systems that support batching, inferring scores for multiple candidate
 *    tokens per beam entry (or even multiple beam entries) in parallel can
 *    significantly improve decoding performance.
 * 3. To conserve memory, autoregressive state should be carefully cleared after
 *    it has been used. State can be cached by mapping it to the beam index in
 *    which it was first conditioned upon for incremental scores, then cleared
 *    after it was used for [batched] score computation.
 */
using EmittingModelUpdateFunc = std::function<
    std::pair<
        std::vector<std::vector<float>>, // A distribution of scores over tokens
                                         // in the token set (inner vector) for
                                         // each candidate in the beam (outer
                                         // vector). This vector must have as
                                         // many elements as there are
                                         // candidates in the beam.
        std::vector<EmittingModelStatePtr>> // A vector of emitting model state;
                                            // each value represents the
                                            // incremental state emitted by
                                            // inference on an autoregressive
                                            // mdoel with a particular token.
                                            // The vector must have as many
                                            // elements as there are candidates
                                            // in the beam.
    (const float*, // Emissions from the input to the autoregressive model
                   // (usually an encoder-like model). Invariant throughout
                   // decoding and set using the
     const int, // N - the size of the token set emitted by the encoder for each
                // time step. Invariant throughout decoding.
     const int, // T - the total number of time steps emitted by the encoder.
                // Invariant throughout decoding.
     const std::vector<int>&, // The raw token ID of the last step for each
                              // element in the beam; the vector has as many
                              // elements as there are candidates in the beam.
     const std::vector<int>&, // The indices of the beam hypotheses for which
                              // the token ID at this index is being proposed.
     const std::vector<EmittingModelStatePtr>&, // State from the previous type
                                                // steps for each candidate in
                                                // the beam. Each hypothesis in
                                                // the beam has its own previous
                                                // state because it required a
                                                // distinct autoregressive input
                                                // to the emitting model; the
                                                // vector has as many elements
                                                // as there are candidates in
                                                // the beam.
     int& // The current time step being decoded -- 0 --> T
     )>;

/* ===================== Candidate-related operations ===================== */

template <class DecoderState>
void candidatesReset(
    double& candidatesBestScore,
    std::vector<DecoderState>& candidates,
    std::vector<DecoderState*>& candidatePtrs) {
  candidatesBestScore = kNegativeInfinity;
  candidates.clear();
  candidatePtrs.clear();
}

template <class DecoderState, class... Args>
void candidatesAdd(
    std::vector<DecoderState>& candidates,
    double& candidatesBestScore,
    const double beamThreshold,
    const double score,
    const Args&... args) {
  if (score >= candidatesBestScore) {
    candidatesBestScore = score;
  }
  if (score >= candidatesBestScore - beamThreshold) {
    candidates.emplace_back(score, args...);
  }
}

template <class DecoderState>
void candidatesStore(
    std::vector<DecoderState>& candidates,
    std::vector<DecoderState*>& candidatePtrs,
    std::vector<DecoderState>& outputs,
    const int beamSize,
    const double threshold,
    const bool logAdd,
    const bool returnSorted) {
  outputs.clear();
  if (candidates.empty()) {
    return;
  }

  /* 1. Select valid candidates */
  for (auto& candidate : candidates) {
    if (candidate.score >= threshold) {
      candidatePtrs.emplace_back(&candidate);
    }
  }

  /* 2. Merge candidates */
  std::sort(
      candidatePtrs.begin(),
      candidatePtrs.end(),
      [](const DecoderState* node1, const DecoderState* node2) {
        int cmp = node1->compareNoScoreStates(node2);
        return cmp == 0 ? node1->score > node2->score : cmp > 0;
      });

  int nHypAfterMerging = 1;
  for (int i = 1; i < candidatePtrs.size(); i++) {
    if (candidatePtrs[i]->compareNoScoreStates(
            candidatePtrs[nHypAfterMerging - 1]) != 0) {
      // Distinct candidate
      candidatePtrs[nHypAfterMerging] = candidatePtrs[i];
      nHypAfterMerging++;
    } else {
      // Same candidate
      double maxScore = std::max(
          candidatePtrs[nHypAfterMerging - 1]->score, candidatePtrs[i]->score);
      if (logAdd) {
        double minScore = std::min(
            candidatePtrs[nHypAfterMerging - 1]->score,
            candidatePtrs[i]->score);
        candidatePtrs[nHypAfterMerging - 1]->score =
            maxScore + std::log1p(std::exp(minScore - maxScore));
      } else {
        candidatePtrs[nHypAfterMerging - 1]->score = maxScore;
      }
    }
  }
  candidatePtrs.resize(nHypAfterMerging);

  /* 3. Sort and prune */
  auto compareNodeScore = [](const DecoderState* node1,
                             const DecoderState* node2) {
    return node1->score > node2->score;
  };

  int nValidHyp = candidatePtrs.size();
  int finalSize = std::min(nValidHyp, beamSize);
  if (!returnSorted && nValidHyp > beamSize) {
    std::nth_element(
        candidatePtrs.begin(),
        candidatePtrs.begin() + finalSize,
        candidatePtrs.begin() + nValidHyp,
        compareNodeScore);
  } else if (returnSorted) {
    std::partial_sort(
        candidatePtrs.begin(),
        candidatePtrs.begin() + finalSize,
        candidatePtrs.begin() + nValidHyp,
        compareNodeScore);
  }

  for (int i = 0; i < finalSize; i++) {
    outputs.emplace_back(std::move(*candidatePtrs[i]));
  }
}

/* ===================== Result-related operations ===================== */

template <class DecoderState>
DecodeResult getHypothesis(const DecoderState* node, const int finalFrame) {
  const DecoderState* node_ = node;
  if (!node_) {
    return DecodeResult();
  }

  DecodeResult res(finalFrame + 1);
  res.score = node_->score;
  res.emittingModelScore = node_->emittingModelScore;
  res.lmScore = node_->lmScore;

  int i = 0;
  while (node_) {
    res.words[finalFrame - i] = node_->getWord();
    res.tokens[finalFrame - i] = node_->token;
    node_ = node_->parent;
    i++;
  }

  return res;
}

template <class DecoderState>
std::vector<DecodeResult> getAllHypothesis(
    const std::vector<DecoderState>& finalHyps,
    const int finalFrame) {
  int nHyp = finalHyps.size();

  std::vector<DecodeResult> res(nHyp);

  for (int r = 0; r < nHyp; r++) {
    const DecoderState* node = &finalHyps[r];
    res[r] = getHypothesis(node, finalFrame);
  }

  return res;
}

template <class DecoderState>
const DecoderState* findBestAncestor(
    const std::vector<DecoderState>& finalHyps,
    int& lookBack) {
  int nHyp = finalHyps.size();
  if (nHyp == 0) {
    return nullptr;
  }

  double bestScore = finalHyps.front().score;
  const DecoderState* bestNode = finalHyps.data();
  for (int r = 1; r < nHyp; r++) {
    const DecoderState* node = &finalHyps[r];
    if (node->score > bestScore) {
      bestScore = node->score;
      bestNode = node;
    }
  }

  int n = 0;
  while (bestNode && n < lookBack) {
    n++;
    bestNode = bestNode->parent;
  }

  const int maxLookBack = lookBack + kLookBackLimit;
  while (bestNode) {
    // Check for first emitted word.
    if (bestNode->isComplete()) {
      break;
    }

    n++;
    bestNode = bestNode->parent;

    if (n == maxLookBack) {
      break;
    }
  }

  lookBack = n;
  return bestNode;
}

template <class DecoderState>
void pruneAndNormalize(
    std::unordered_map<int, std::vector<DecoderState>>& hypothesis,
    const int startFrame,
    const int lookBack) {
  /* 1. Move things from back of hypothesis to front. */
  for (int i = 0; i < hypothesis.size(); i++) {
    if (i <= lookBack) {
      hypothesis[i].swap(hypothesis[i + startFrame]);
    } else {
      hypothesis[i].clear();
    }
  }

  /* 2. Avoid further back-tracking */
  for (DecoderState& hyp : hypothesis[0]) {
    hyp.parent = nullptr;
  }

  /* 3. Avoid score underflow/overflow. */
  double largestScore = hypothesis[lookBack].front().score;
  for (int i = 1; i < hypothesis[lookBack].size(); i++) {
    if (largestScore < hypothesis[lookBack][i].score) {
      largestScore = hypothesis[lookBack][i].score;
    }
  }

  for (int i = 0; i < hypothesis[lookBack].size(); i++) {
    hypothesis[lookBack][i].score -= largestScore;
  }
}

/* ===================== LM-related operations ===================== */

template <class DecoderState>
void updateLMCache(const LMPtr& lm, std::vector<DecoderState>& hypothesis) {
  // For ConvLM update cache
  std::vector<LMStatePtr> states;
  for (const auto& hyp : hypothesis) {
    states.emplace_back(hyp.lmState);
  }
  lm->updateCache(states);
}
} // namespace text
} // namespace lib
} // namespace fl
