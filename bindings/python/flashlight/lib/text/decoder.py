#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT-style license found in the
LICENSE file in the root directory of this source tree.
"""

import logging

from .flashlight_lib_text_decoder import (
    CriterionType,
    DecodeResult,
    LexiconDecoder,
    LexiconDecoderOptions,
    LexiconFreeDecoder,
    LexiconFreeDecoderOptions,
    LM,
    LMState,
    SmearingMode,
    Trie,
    TrieNode,
)

try:
    from .flashlight_lib_text_decoder import KenLM
except ImportError:
    logging.info("Flashlight Text Python bindings installed without KenLM.")
