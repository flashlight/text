/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <numeric>

#include "flashlight/lib/text/decoder/LexiconFreeSeq2SeqDecoder.h"

namespace fl {
namespace lib {
namespace text {

void LexiconFreeSeq2SeqDecoder::decodeStep(
    const float* emissions,
    int T,
    int N) {
  // Extend hyp_ buffer
  if (hyp_.size() < maxOutputLength_ + 2) {
    for (int i = hyp_.size(); i < maxOutputLength_ + 2; i++) {
      hyp_.emplace(i, std::vector<LexiconFreeSeq2SeqDecoderState>());
    }
  }

  // Start from here.
  hyp_[0].clear();
  hyp_[0].emplace_back(0.0, lm_->start(false), nullptr, -1, nullptr);

  // Decode frame by frame
  int t = 0;
  for (; t < maxOutputLength_; t++) {
    candidatesReset(candidatesBestScore_, candidates_, candidatePtrs_);

    // Batch forwarding - computing scores from the autoregressive model
    rawY_.clear();
    rawBeamIdx_.clear();
    rawPrevStates_.clear();
    for (const LexiconFreeSeq2SeqDecoderState& prevHyp : hyp_[t]) {
      const EmittingModelStatePtr& prevState = prevHyp.emittingModelState;
      if (prevHyp.token == eos_) {
        continue;
      }
      rawY_.push_back(prevHyp.token);
      rawBeamIdx_.push_back(prevHyp.prevHypIdx.value_or(-1));
      rawPrevStates_.push_back(prevState);
    }
    if (rawY_.size() == 0) {
      break;
    }

    std::vector<std::vector<float>> emittingModelScores;
    std::vector<EmittingModelStatePtr> outStates;
    std::tie(emittingModelScores, outStates) = emittingModelUpdateFunc_(
        emissions, N, T, rawY_, rawBeamIdx_, rawPrevStates_, t);

    std::vector<size_t> idx(emittingModelScores.back().size());

    // Generate new hypothesis
    for (int hypo = 0, validHypo = 0; hypo < hyp_[t].size(); hypo++) {
      const LexiconFreeSeq2SeqDecoderState& prevHyp = hyp_[t][hypo];
      // Change nothing for completed hypothesis
      if (prevHyp.token == eos_) {
        candidatesAdd(
            candidates_,
            candidatesBestScore_,
            opt_.beamThreshold,
            prevHyp.score,
            prevHyp.lmState,
            &prevHyp,
            eos_,
            nullptr,
            prevHyp.emittingModelScore,
            prevHyp.lmScore,
            hypo);
        continue;
      }

      const EmittingModelStatePtr& outState = outStates[validHypo];
      if (!outState) {
        validHypo++;
        continue;
      }

      std::iota(idx.begin(), idx.end(), 0);
      if (emittingModelScores[validHypo].size() > opt_.beamSizeToken) {
        std::partial_sort(
            idx.begin(),
            idx.begin() + opt_.beamSizeToken,
            idx.end(),
            [&emittingModelScores, &validHypo](
                const size_t& l, const size_t& r) {
              return emittingModelScores[validHypo][l] >
                  emittingModelScores[validHypo][r];
            });
      }

      for (int r = 0; r <
           std::min(emittingModelScores[validHypo].size(),
                    (size_t)opt_.beamSizeToken);
           r++) {
        int n = idx[r];
        double emittingModelScore = emittingModelScores[validHypo][n];

        if (n == eos_) { /* (1) Try eos */
          auto lmStateScorePair = lm_->finish(prevHyp.lmState);
          auto lmScore = lmStateScorePair.second;

          candidatesAdd(
              candidates_,
              candidatesBestScore_,
              opt_.beamThreshold,
              prevHyp.score + emittingModelScore + opt_.eosScore +
                  opt_.lmWeight * lmScore,
              lmStateScorePair.first,
              &prevHyp,
              n,
              nullptr,
              prevHyp.emittingModelScore + emittingModelScore,
              prevHyp.lmScore + lmScore,
              hypo);
        } else { /* (2) Try normal token */
          auto lmStateScorePair = lm_->score(prevHyp.lmState, n);
          auto lmScore = lmStateScorePair.second;
          candidatesAdd(
              candidates_,
              candidatesBestScore_,
              opt_.beamThreshold,
              prevHyp.score + emittingModelScore + opt_.lmWeight * lmScore,
              lmStateScorePair.first,
              &prevHyp,
              n,
              outState,
              prevHyp.emittingModelScore + emittingModelScore,
              prevHyp.lmScore + lmScore,
              hypo);
        }
      }
      validHypo++;
    }
    candidatesStore(
        candidates_,
        candidatePtrs_,
        hyp_[t + 1],
        opt_.beamSize,
        candidatesBestScore_ - opt_.beamThreshold,
        opt_.logAdd,
        true);
    updateLMCache(lm_, hyp_[t + 1]);
  } // End of decoding

  while (t > 0 && hyp_[t].empty()) {
    --t;
  }
  hyp_[maxOutputLength_ + 1].resize(hyp_[t].size());
  for (int i = 0; i < hyp_[t].size(); i++) {
    hyp_[maxOutputLength_ + 1][i] = std::move(hyp_[t][i]);
  }
}

std::vector<DecodeResult> LexiconFreeSeq2SeqDecoder::getAllFinalHypothesis()
    const {
  return getAllHypothesis(hyp_.find(maxOutputLength_ + 1)->second, hyp_.size());
}

DecodeResult LexiconFreeSeq2SeqDecoder::getBestHypothesis(
    int /* unused */) const {
  return getHypothesis(
      hyp_.find(maxOutputLength_ + 1)->second.data(), hyp_.size());
}

void LexiconFreeSeq2SeqDecoder::prune(int /* unused */) {
  return;
}

int LexiconFreeSeq2SeqDecoder::nDecodedFramesInBuffer() const {
  /* unused function */
  return -1;
}
} // namespace text
} // namespace lib
} // namespace fl
