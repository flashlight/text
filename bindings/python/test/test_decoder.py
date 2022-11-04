#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT-style license found in the
LICENSE file in the root directory of this source tree.
"""

# Perform beam-search decoding with word-level LM
# this is test with dumped emitting model scores

import math
import os
import pickle
import struct
import unittest

import numpy as np
from flashlight.lib.text.decoder import (
    CriterionType,
    LexiconDecoder,
    LexiconDecoderOptions,
    LexiconFreeDecoder,
    LexiconFreeDecoderOptions,
    SmearingMode,
    Trie,
    ZeroLM,
)
from flashlight.lib.text.dictionary import (
    create_word_dict,
    Dictionary,
    load_words,
    pack_replabels,
)


def read_struct(file, fmt):
    return struct.unpack(fmt, file.read(struct.calcsize(fmt)))


def load_tn(path):
    """
    Load time size and number of tokens from the dump
    (defines the score to move from token_i to token_j)

    Returns:
    --------
    int, int
    """
    with open(path, "rb") as file:
        T = read_struct(file, "i")[0]
        N = read_struct(file, "i")[0]
        return T, N


def load_emissions(path, T, N):
    """
    Load precomputed transition matrix
    (defines the score to move from token_i to token_j)

    Returns:
    --------
    numpy.array of shape [Batch=1, Time, Ntokens]
    """
    with open(path, "rb") as file:
        return np.frombuffer(file.read(T * N * 4), dtype=np.float32)


def load_transitions(path, N):
    """
    Load precomputed transition matrix
    (defines the score to move from token_i to token_j)

    Returns:
    --------
    numpy.array of shape [Ntokens, Ntokens]
    """
    with open(path, "rb") as file:
        return np.frombuffer(file.read(N * N * 4), dtype=np.float32)


def tkn_to_idx(spelling, token_dict, maxReps=0):
    result = []
    for token in spelling:
        result.append(token_dict.get_index(token))
    return pack_replabels(result, token_dict, maxReps)


class DecoderTestCase(unittest.TestCase):
    def test(self):
        if os.getenv("USE_KENLM", "OFF").upper() in ["OFF", "0", "NO", "FALSE", "N"]:
            self.skipTest("KenLM required to run decoder test.")

        from flashlight.lib.text.decoder import KenLM

        data_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            os.getenv(
                "DATA_DIR",
                default="../../../flashlight/lib/text/test/decoder/data",
            ),
        )
        if data_path is None:
            raise FileNotFoundError(
                "DATA_DIR environment variable must be set to the "
                "path containing decoder data test components"
            )
        print(f"Running {__class__} with DATA_DIR={data_path}")

        # load test files
        # load time and number of tokens for dumped emitting model scores
        T, N = load_tn(os.path.join(data_path, "TN.bin"))
        # load emissions [Batch=1, Time, Ntokens]
        emissions = load_emissions(os.path.join(data_path, "emission.bin"), T, N)
        # load transitions (from ASG loss optimization) [Ntokens, Ntokens]
        transitions = load_transitions(os.path.join(data_path, "transition.bin"), N)
        # load lexicon file, which defines spelling of words
        # the format word and its tokens spelling separated by the spaces,
        # for example for letters tokens with ASG loss:
        # ann a n 1 |
        lexicon = load_words(os.path.join(data_path, "words.lst"))
        # read lexicon and store it in the w2l dictionary
        word_dict = create_word_dict(lexicon)
        # create w2l dict with tokens set (letters in this example)
        token_dict = Dictionary(os.path.join(data_path, "letters.lst"))
        # add repetition symbol as soon as we have ASG emitting model
        token_dict.add_entry("<1>")
        # create Kenlm language model
        lm = KenLM(os.path.join(data_path, "lm.arpa"), word_dict)

        # test LM
        sentence = ["the", "cat", "sat", "on", "the", "mat"]
        # start LM with nothing, get its current state
        lm_state = lm.start(False)
        total_score = 0
        lm_score_target = [-1.05971, -4.19448, -3.33383, -2.76726, -1.16237, -4.64589]
        # iterate over words in the sentence
        for i in range(len(sentence)):
            # score lm, taking current state and index of the word
            # returns new state and score for the word
            lm_state, lm_score = lm.score(lm_state, word_dict.get_index(sentence[i]))
            self.assertAlmostEqual(lm_score, lm_score_target[i], places=4)
            # add score of the current word to the total sentence score
            total_score += lm_score
        # move lm to the final state, the score returned is for eos
        lm_state, lm_score = lm.finish(lm_state)
        total_score += lm_score
        self.assertAlmostEqual(total_score, -19.5123, places=4)

        # build trie
        # Trie is necessary to do beam-search decoding with word-level lm
        # We restrict our search only to the words from the lexicon
        # Trie is constructed from the lexicon, each node is a token
        # path from the root to a leaf corresponds to a word spelling in the lexicon

        # get separator index
        separator_idx = token_dict.get_index("|")
        # get unknown word index
        unk_idx = word_dict.get_index("<unk>")
        # create the trie, specifying how many tokens we have and silence index
        trie = Trie(token_dict.index_size(), separator_idx)
        start_state = lm.start(False)

        # use heuristic for the trie, called smearing:
        # predict lm score for each word in the lexicon, set this score to a leaf
        # (we predict lm score for each word as each word starts a sentence)
        # word score of a leaf is propagated up to the root to have some proxy score
        # for any intermediate path in the trie
        # SmearingMode defines the function how to process scores
        # in a node came from the children nodes:
        # could be max operation or logadd or none
        for word, spellings in lexicon.items():
            usr_idx = word_dict.get_index(word)
            _, score = lm.score(start_state, usr_idx)
            for spelling in spellings:
                # max_reps should be 1; using 0 here to match DecoderTest bug
                spelling_idxs = tkn_to_idx(spelling, token_dict, 1)
                trie.insert(spelling_idxs, usr_idx, score)

        trie.smear(SmearingMode.MAX)

        # check that trie is built in consistency with c++
        trie_score_target = [-1.05971, -2.87742, -2.64553, -3.05081, -1.05971, -3.08968]
        for i in range(len(sentence)):
            word = sentence[i]
            # max_reps should be 1; using 0 here to match DecoderTest bug
            word_tensor = tkn_to_idx([c for c in word], token_dict, 1)
            node = trie.search(word_tensor)
            self.assertAlmostEqual(node.max_score, trie_score_target[i], places=4)

        # Define decoder options:
        # LexiconDecoderOptions (beam_size, token_beam_size, beam_threshold, lm_weight,
        #                 word_score, unk_score, sil_score,
        #                 log_add, criterion_type (ASG or CTC))
        opts = LexiconDecoderOptions(
            beam_size=2500,
            beam_size_token=25000,
            beam_threshold=100.0,
            lm_weight=2.0,
            word_score=2.0,
            unk_score=-math.inf,
            sil_score=-1,
            log_add=False,
            criterion_type=CriterionType.ASG,
        )

        # define lexicon beam-search decoder with word-level lm
        # LexiconDecoder(decoder options, trie, lm, silence index,
        #                blank index (for CTC), unk index,
        #                transitions matrix, is token-level lm)
        decoder = LexiconDecoder(
            options=opts,
            trie=trie,
            lm=lm,
            sil_token_idx=separator_idx,
            blank_token_idx=-1,
            unk_token_idx=unk_idx,
            transitions=transitions,
            is_token_lm=False,
        )

        # run decoding
        # decoder.decode(emissions, Time, Ntokens)
        # result is a list of sorted hypothesis, 0-index is the best hypothesis
        # each hypothesis is a struct with "score" and "words" representation
        # in the hypothesis and the "tokens" representation
        results = decoder.decode(emissions.ctypes.data, T, N)

        print(f"Decoding complete, obtained {len(results)} results")
        print("Showing top 5 results:")
        for i in range(min(5, len(results))):
            prediction = []
            for idx in results[i].tokens:
                if idx < 0:
                    break
                prediction.append(token_dict.get_entry(idx))
            prediction = " ".join(prediction)
            print(
                f"score={results[i].score} emittingModelScore={results[i].emittingModelScore} lmScore={results[i].lmScore} prediction='{prediction}'"
            )

        self.assertEqual(len(results), 16)
        hyp_score_target = [-284.0998, -284.108, -284.119, -284.127, -284.296]
        for i in range(min(5, len(results))):
            self.assertAlmostEqual(results[i].score, hyp_score_target[i], places=3)


class DecoderPickleTestCase(unittest.TestCase):
    def cmp_options(self, lhs, rhs):
        self.assertEqual(lhs.beam_size, rhs.beam_size)
        self.assertEqual(lhs.beam_size_token, rhs.beam_size_token)
        self.assertEqual(lhs.beam_threshold, rhs.beam_threshold)
        self.assertEqual(lhs.lm_weight, rhs.lm_weight)
        self.assertEqual(lhs.sil_score, rhs.sil_score)
        self.assertEqual(lhs.log_add, rhs.log_add)
        self.assertEqual(lhs.criterion_type, rhs.criterion_type)

    def test(self):
        beam_size = 3
        beam_size_token = 5
        beam_threshold = 10.0
        lm_weight = 7.0
        word_score = 3.0
        unk_score = 2.0
        sil_score = 5.0
        log_add = True
        criterion_type = CriterionType.CTC
        lm = ZeroLM()
        sil_token_idx = 4
        blank_token_idx = 22
        transitions = [1.0, 4.0, 5.0, 9.0]

        # ]----- non-autoregressive decoding
        # lexicon-free
        lex_free_opts = LexiconFreeDecoderOptions(
            beam_size=beam_size,
            beam_size_token=beam_size_token,
            beam_threshold=beam_threshold,
            lm_weight=lm_weight,
            sil_score=sil_score,
            log_add=log_add,
            criterion_type=criterion_type,
        )
        pickled = pickle.dumps(lex_free_opts)
        _lex_free_opts = pickle.loads(pickled)
        self.cmp_options(_lex_free_opts, lex_free_opts)

        lex_free_decoder = LexiconFreeDecoder(
            options=lex_free_opts,
            lm=lm,
            sil_token_idx=sil_token_idx,
            blank_token_idx=blank_token_idx,
            transitions=transitions,
        )
        pickled = pickle.dumps(lex_free_decoder)
        _lex_free_decoder = pickle.loads(pickled)
        self.assertEqual(
            _lex_free_decoder.get_sil_idx(), lex_free_decoder.get_sil_idx()
        )
        self.assertEqual(
            _lex_free_decoder.get_blank_idx(), lex_free_decoder.get_blank_idx()
        )
        self.assertEqual(
            _lex_free_decoder.get_transitions(), lex_free_decoder.get_transitions()
        )
        self.cmp_options(
            _lex_free_decoder.get_options(), lex_free_decoder.get_options()
        )

        # lexicon-based
        lex_ops = LexiconDecoderOptions(
            beam_size=beam_size,
            beam_size_token=beam_size_token,
            beam_threshold=beam_threshold,
            lm_weight=lm_weight,
            word_score=word_score,
            unk_score=unk_score,
            sil_score=sil_score,
            log_add=log_add,
            criterion_type=criterion_type,
        )
        pickled = pickle.dumps(lex_ops)
        _lex_ops = pickle.loads(pickled)
        self.assertEqual(_lex_ops.beam_size, lex_ops.beam_size)
        self.assertEqual(_lex_ops.beam_size_token, lex_ops.beam_size_token)
        self.assertEqual(_lex_ops.beam_threshold, lex_ops.beam_threshold)
        self.assertEqual(_lex_ops.lm_weight, lex_ops.lm_weight)
        self.assertEqual(_lex_ops.word_score, lex_ops.word_score)
        self.assertEqual(_lex_ops.unk_score, lex_ops.unk_score)
        self.assertEqual(_lex_ops.sil_score, lex_ops.sil_score)
        self.assertEqual(_lex_ops.log_add, lex_ops.log_add)
        self.assertEqual(_lex_ops.criterion_type, lex_ops.criterion_type)
        # serialization of lexicon_decoder is currently unsupported


if __name__ == "__main__":
    unittest.main()
