#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT-style license found in the
LICENSE file in the root directory of this source tree.
"""

import importlib.util
import os
import site
import sys
import textwrap

if sys.platform == "win32":
    # KenLM is installed directly in site-packages
    for path in site.getsitepackages():
        os.add_dll_directory(path)

# Check if the KenLM Python extension (flashlight_lib_text_decoder_kenlm) is
# available. If it is not, bindings were built without KenLM support.
if (
    importlib.util.find_spec("flashlight.lib.text.flashlight_lib_text_decoder_kenlm")
    is None
):
    raise ImportError(
        textwrap.dedent(
            """
            Installed Flashlight Text Python bindings were built without KenLM
            support. Install the bindings from PyPI, or set the environment
            variable USE_KENLM=1 if building the bindings from source.
            """
        )
    )


try:
    from ..flashlight_lib_text_decoder_kenlm import (  # @manual=fbcode//deeplearning/projects/flashlight-text/bindings/python:flashlight_lib_text_decoder_kenlm
        KenLM,
    )
except ImportError:
    raise ImportError(
        textwrap.dedent(
            """
            Could not import the Flashlight Text KenLM extension. KenLM is
            required to use the Flashlight Text decoder with KenLM support.
            Install KenLM with:

            pip install git+https://github.com/kpu/kenlm
            """
        )
    )
