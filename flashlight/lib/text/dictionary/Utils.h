/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "flashlight/lib/text/Defines.h"
#include "flashlight/lib/text/dictionary/Dictionary.h"

namespace fl {
namespace lib {
namespace text {

using LexiconMap =
    std::unordered_map<std::string, std::vector<std::vector<std::string>>>;

FL_TEXT_API Dictionary createWordDict(const LexiconMap& lexicon);

FL_TEXT_API LexiconMap
loadWords(const std::string& filename, int maxWords = -1);

// split word into tokens abc -> {"a", "b", "c"}
// Works with ASCII, UTF-8 encodings
FL_TEXT_API std::vector<std::string> splitWrd(const std::string& word);

/**
 * Pack a token sequence by replacing consecutive repeats with replabels,
 * e.g. "abbccc" -> "ab1c2". The tokens "1", "2", ..., `to_string(maxReps)`
 * must already be in `dict`.
 */
FL_TEXT_API std::vector<int> packReplabels(
    const std::vector<int>& tokens,
    const Dictionary& dict,
    int maxReps);

/**
 * Unpack a token sequence by replacing replabels with repeated tokens,
 * e.g. "ab1c2" -> "abbccc". The tokens "1", "2", ..., `to_string(maxReps)`
 * must already be in `dict`.
 */
FL_TEXT_API std::vector<int> unpackReplabels(
    const std::vector<int>& tokens,
    const Dictionary& dict,
    int maxReps);

/**
 * Map the spelling of a word to letter indices as defined by a Dictionary
 * with a maximum number of replabels as defined.
 */
FL_TEXT_API std::vector<int> tkn2Idx(
    const std::vector<std::string>& spelling,
    const fl::lib::text::Dictionary& tokenDict,
    int maxReps);

} // namespace text
} // namespace lib
} // namespace fl
