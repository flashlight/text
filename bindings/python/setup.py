#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT-style license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import platform
import re
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

# Environment variables:
# - `USE_KENLM=0` disables building KenLM
# By default build with USE_KENLM=1


def check_env_flag(name, default=""):
    return os.getenv(name, default).upper() in ["ON", "1", "YES", "TRUE", "Y"]


def check_negative_env_flag(name, default=""):
    return os.getenv(name, default).upper() in ["OFF", "0", "NO", "FALSE", "N"]


class CMakeExtension(Extension):
    def __init__(self, name):
        Extension.__init__(self, name, sources=[])


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        cmake_version = re.search(r"version\s*([\d.]+)", out.decode().lower()).group(1)
        cmake_version_tuple = tuple([int(v) for v in cmake_version.split(".")])
        if cmake_version_tuple < (3, 16):
            raise RuntimeError(
                f"CMake >= 3.16 is required to build flashlight-text; found {cmake_version}"
            )

        # our CMakeLists builds all the extensions at once
        for ext in self.extensions:
            self.build_extensions(ext)

    def build_extensions(self, ext):
        ext_dir = Path(self.get_ext_fullpath(ext.name)).absolute()
        while ext_dir.name != "flashlight":
            ext_dir = ext_dir.parent
        ext_dir = str(ext_dir.parent)
        source_dir = str(Path(__file__).absolute().parent.parent.parent)
        use_kenlm = "OFF" if check_negative_env_flag("USE_KENLM") else "ON"
        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + ext_dir,
            "-DPYTHON_EXECUTABLE=" + sys.executable,
            "-DBUILD_SHARED_LIBS=ON",
            "-DFL_TEXT_BUILD_STANDALONE=OFF",
            "-DFL_TEXT_BUILD_TESTS=OFF",
            "-DFL_TEXT_BUILD_PYTHON=ON",
            "-DFL_TEXT_USE_KENLM=" + use_kenlm,
        ]
        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        if platform.system() == "Windows":
            cmake_args += [
                "-DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=ON",
                "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), ext_dir),
                "-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), ext_dir),
            ]
            if sys.maxsize > 2**32:
                cmake_args += ["-A", "x64"]
            build_args += ["--", "/m"]
        else:
            cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
            build_args += ["--", "-j4"]

        env = os.environ.copy()
        env["CXXFLAGS"] = '{} -fPIC -DVERSION_INFO=\\"{}\\"'.format(
            env.get("CXXFLAGS", ""), self.distribution.get_version()
        )

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(
            ["cmake", source_dir] + cmake_args, cwd=self.build_temp, env=env
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )


setup(
    name="flashlight-text",
    version="0.1",
    author="Flashlight Contributors",
    author_email="oncall+fair_speech@xmail.facebook.com",
    description="Flashlight Text bindings for python",
    long_description="",
    packages=["flashlight.lib.text"],
    ext_modules=[
        CMakeExtension("flashlight.lib.text.decoder"),
        CMakeExtension("flashlight.lib.text.dictionary"),
    ],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    license="BSD licensed, as found in the LICENSE file",
    python_requires=">=3.6",
)
