#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT-style license found in the
LICENSE file in the root directory of this source tree.
"""

name = "text"

try:
    from .version import __version__  # noqa: F401
except ImportError:
    __version__ = "0.0.0"
