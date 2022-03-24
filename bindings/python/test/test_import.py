"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT-style license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import unittest
import logging


class ImportTestCase(unittest.TestCase):
    def test_import_lib_text(self):
        from flashlight.lib.text import dictionary as fl_dict
        from flashlight.lib.text import decoder as fl_decoder

        if os.getenv("USE_KENLM", "OFF").upper() not in [
            "OFF",
            "0",
            "NO",
            "FALSE",
            "N",
        ]:
            from flashlight.lib.text.decoder import KenLM
        else:
            logging.info("Flashlight Text bindings built without KenLM")


if __name__ == "__main__":
    unittest.main()
