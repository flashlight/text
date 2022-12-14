#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT-style license found in the
LICENSE file in the root directory of this source tree.
"""

import logging

from .flashlight_lib_text_decoder import (
    create_emitting_model_state,
    CriterionType,
    DecodeResult,
    EmittingModelState,
    get_obj_from_emitting_model_state,
    LexiconDecoder,
    LexiconDecoderOptions,
    LexiconFreeDecoder,
    LexiconFreeDecoderOptions,
    LexiconFreeSeq2SeqDecoder,
    LexiconFreeSeq2SeqDecoderOptions,
    LexiconSeq2SeqDecoder,
    LexiconSeq2SeqDecoderOptions,
    LM,
    LMState,
    SmearingMode,
    Trie,
    TrieNode,
    ZeroLM,
)

try:
    from .flashlight_lib_text_decoder import KenLM
except ImportError:
    logging.info("Flashlight Text Python bindings installed without KenLM.")
