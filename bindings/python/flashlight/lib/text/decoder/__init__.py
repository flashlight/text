#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT-style license found in the
LICENSE file in the root directory of this source tree.
"""

import textwrap
import warnings

from ..flashlight_lib_text_decoder import (  # noqa  # @manual=fbcode//deeplearning/projects/flashlight-text/bindings/python:flashlight_lib_text_decoder
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

_KENLM_CLASSNAME = "KenLM"


def __getattr__(name: str):
    """
    For backwards compatibility, still allow
    `from flashlight.lib.text.decoder import KenLM` to succeed. This import
    path will eventually be deprecated.
    """
    if name == _KENLM_CLASSNAME:
        warnings.warn(
            textwrap.dedent(
                """
        Flashlight Text's KenLM cls has been moved - this approach to importing
        it will be deprecated. Use the following new path to import:

        from flashlight.lib.text.decoder.kenlm import KenLM
        """
            ),
            PendingDeprecationWarning,
        )

        from .kenlm import KenLM

        globals()[_KENLM_CLASSNAME] = KenLM

        return KenLM
    raise AttributeError(f"module {__name__} has no attribute {name}")
