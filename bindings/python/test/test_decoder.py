#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT-style license found in the
LICENSE file in the root directory of this source tree.
"""


import math
import os
import pickle
import struct
import sys
import unittest
from dataclasses import dataclass

import numpy as np
from flashlight.lib.text.decoder import (
    create_emitting_model_state,
    CriterionType,
    get_obj_from_emitting_model_state,
    LexiconDecoder,
    LexiconDecoderOptions,
    LexiconFreeDecoder,
    LexiconFreeDecoderOptions,
    LexiconFreeSeq2SeqDecoder,
    LexiconFreeSeq2SeqDecoderOptions,
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
    @unittest.skipIf(
        sys.platform.startswith("win"),
        "KenLM .arpa files need further debugging on Windows",
    )
    def test(self):
        if os.getenv("USE_KENLM", "OFF").upper() in ["OFF", "0", "NO", "FALSE", "N"]:
            self.skipTest("KenLM required to run decoder test.")

        from flashlight.lib.text.decoder.kenlm import KenLM

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


@dataclass
class Seq2SeqTestModelState:
    """
    A simulation of model state. These are synthetically created for the test
    but store information about model scores for the next timestep (i.e.
    "hidden states" in an autoregressive sense)
    """

    timestep: int
    token_idx: int
    score: float

    def __init__(self, timestep, token_idx, score):
        self.timestep = timestep
        self.token_idx = token_idx
        self.score = score


class DecoderLexiconFreeSeq2SeqTestCase(unittest.TestCase):
    def update_func(
        self,
        emissions_ptr,
        N,
        T,
        prev_step_token_idxs,
        prev_step_beam_idxs,
        prev_step_model_states,
        timestep,
    ):
        """
        For the purposes of testing, this closure is a method on the test class
        so it has access to common data that can be internally compared against.
        Practically, this will be a standalone method that captures other state
        (i.e. autoregressive models, configurations, reusable buffers, etc).
        """
        self.assertEqual(N, self.N)
        self.assertEqual(T, self.T)
        self.assertEqual(len(prev_step_token_idxs), len(prev_step_model_states))
        if timestep == 0:
            self.assertEqual(prev_step_token_idxs, [-1])
            self.assertEqual(prev_step_beam_idxs, [-1])
            # This obj is actually a nullptr internally -- do not use
            self.assertEqual(len(prev_step_model_states), 1)
        else:
            for _prev_state in prev_step_model_states:
                prev_state = get_obj_from_emitting_model_state(_prev_state)
                if timestep == 1:
                    self.assertEqual(prev_state.score, -1)
                    self.assertEqual(prev_state.token_idx, 0)
                    self.assertEqual(
                        prev_step_beam_idxs, [0] * len(prev_step_token_idxs)
                    )
                else:
                    self.assertGreater(timestep, 1)
                    scores = self.model_score_mapping[timestep - 1]
                    max_score = max(scores)
                    self.assertAlmostEqual(prev_state.score, max_score)
                    self.assertEqual(prev_state.token_idx, scores.index(max_score))
                    self.assertEqual(
                        prev_step_beam_idxs, [0] * len(prev_step_token_idxs)
                    )
        cur_model_score = self.model_score_mapping[timestep]

        model_states = []
        for i, _ in enumerate(prev_step_token_idxs):
            model_states.append(
                create_emitting_model_state(
                    Seq2SeqTestModelState(
                        timestep=timestep,
                        token_idx=i,
                        score=(-1 if timestep == 0 else cur_model_score[i]),
                    )
                )
            )

        out_probs = [cur_model_score] * len(prev_step_token_idxs)
        return out_probs, model_states

    def test(self):
        self.T = T = 3
        self.N = N = 4
        self.emissions = np.array([i - (T * N) / 2 for i in range(0, T * N)])
        self.assertEqual(len(self.emissions), T * N)

        beam_size = 2
        eos_idx = 4
        max_output_length = 3

        # timestep --> autoregressive scores
        self.model_score_mapping = {
            0: [0.1, 0.1, 0.5, 0.1],
            1: [0.5, 0.2, 0.1, 0.0],
            2: [0.1, 0.5, 0.1, 0.1],
        }
        self.assertEqual(len(self.model_score_mapping), T)

        options = LexiconFreeSeq2SeqDecoderOptions(
            beam_size=beam_size,
            beam_size_token=4,
            beam_threshold=1000,
            lm_weight=0,
            eos_score=0,
            log_add=True,
        )

        decoder = LexiconFreeSeq2SeqDecoder(
            options=options,
            lm=ZeroLM(),
            eos_idx=eos_idx,
            update_func=self.update_func,
            max_output_length=max_output_length,
        )

        decoder.decode_step(self.emissions.ctypes.data, T, N)
        hyps = decoder.get_all_final_hypothesis()

        # Validate final hypotheses state
        self.assertEqual(len(hyps), 2)
        self.assertAlmostEqual(hyps[0].score, 0.5 + 0.5 + 0.5)
        self.assertAlmostEqual(hyps[1].score, 0.5 + 0.2 + 0.5)

        for hyp in hyps:
            self.assertAlmostEqual(hyp.lmScore, 0)  # ZeroLM
            self.assertAlmostEqual(hyp.score, hyp.emittingModelScore)
            self.assertEqual(len(hyp.words), len(hyp.tokens))  # since lexfree

        self.assertEqual(hyps[0].tokens, [-1, -1, -1, 2, 0, 1])
        self.assertEqual(hyps[1].tokens, [-1, -1, -1, 2, 1, 1])


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


def main() -> None:
    unittest.main()


if __name__ == "__main__":
    main()  # pragma: no cover
