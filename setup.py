#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT-style license found in the
LICENSE file in the root directory of this source tree.
"""

import datetime
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

from setuptools import Extension, find_namespace_packages, setup
from setuptools.command.build_ext import build_ext

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Path relative to project root that contains Python artifacts for packaging
PACKAGE_DIR = "bindings/python"
ARTIFACTS_DIR = os.path.join(PACKAGE_DIR, "flashlight/lib/text")
BUILD_VERSION_PATH = Path(os.path.join(THIS_DIR, "BUILD_VERSION.txt"))
VERSION_TXT_PATH = Path(os.path.join(THIS_DIR, PACKAGE_DIR, "version.txt"))
VERSION_PY_PATH = Path(
    os.path.join(THIS_DIR, PACKAGE_DIR, "flashlight", "lib", "text", "version.py")
)

CMAKE_MINIMUM_VERSION = (3, 18)  # 3.18


# Environment variables:
# - `USE_KENLM=0` disables building KenLM
# By default build with USE_KENLM=1


def check_env_flag(name, default="") -> bool:
    return os.getenv(name, default).upper() in ["ON", "1", "YES", "TRUE", "Y"]


def check_negative_env_flag(name, default="") -> bool:
    return os.getenv(name, default).upper() in ["OFF", "0", "NO", "FALSE", "N"]


def get_local_version_suffix() -> str:
    date_suffix = datetime.datetime.now().strftime("%Y%m%d")
    git_hash = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent
    ).decode("ascii")[:-1]
    return f"+{git_hash}.d{date_suffix}"


def write_version_file(version: str):
    version_path = os.path.join(THIS_DIR, ARTIFACTS_DIR, "version.py")
    with open(version_path, "w") as f:
        f.write("# noqa: C801\n")
        f.write(f'__version__ = "{version}"\n')
        tag = os.getenv("GIT_TAG")
        if tag is not None:
            f.write(f'git_tag = "{tag}"\n')


def get_kenlm_paths(_basedir: str) -> str:
    """
    Download and untar KenLM head - headers are needed if building KenLM bindings.
    """
    base_dir = Path(_basedir) / "kenlm"

    KENLM_SOURCE_DIR = base_dir / "kenlm"
    kenlm_header_path = Path(KENLM_SOURCE_DIR)

    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
        KENLM_TARBALL_URL = "https://github.com/kpu/kenlm/tarball/master"
        KENLM_TARBALL_NAME = "kenlm.tar.gz"
        KENLM_TARBALL_PATH = base_dir / KENLM_TARBALL_NAME

        res = urllib.request.urlretrieve(KENLM_TARBALL_URL, KENLM_TARBALL_PATH)
        if res is None:
            raise RuntimeError(
                "Failed to download KenLM headers. Build with ",
                "environment variable USE_KENLM=OFF to disable building with KenLM support.",
            )

        # Extract the tarfile to the KENLM_DIR
        tar = tarfile.open(KENLM_TARBALL_PATH, "r")
        os.mkdir(KENLM_SOURCE_DIR)
        tar.extractall(path=KENLM_SOURCE_DIR)
        shutil.move(
            KENLM_SOURCE_DIR / os.listdir(KENLM_SOURCE_DIR)[0],
            KENLM_SOURCE_DIR / "kenlm",
        )

    try:
        import kenlm
    except ImportError:
        raise RuntimeError(
            "KenLM is not installed or failed to import. ",
            "Install with `pip install git+https://github.com/kpu/kenlm`. To build ",
            "Flashlight Text bindings without KenLM support, set the environment ",
            "variable USE_KENLM=0.",
        )
    kenlm_lib_path = kenlm.__file__

    return Path(kenlm_header_path), Path(kenlm_lib_path).parent


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
        if cmake_version_tuple < CMAKE_MINIMUM_VERSION:
            raise RuntimeError(
                f"CMake >= 3.18 is required to build flashlight-text; found {cmake_version}"
            )

        # our CMakeLists builds all the extensions at once
        for ext in self.extensions:
            self.build_extensions(ext)

    def build_extensions(self, ext):
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        ext_dir = str(Path(self.get_ext_fullpath(ext.name)).absolute().parent)
        source_dir = str(Path(__file__).absolute().parent)
        use_kenlm = not check_negative_env_flag("USE_KENLM")  # on unless disabled
        kenlm_header_path, kenlm_lib_path = (
            get_kenlm_paths(self.build_temp) if use_kenlm else (None, None)
        )
        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + ext_dir,
            "-DPython3_EXECUTABLE=" + sys.executable,
            "-DBUILD_SHARED_LIBS=ON",
            "-DFL_TEXT_BUILD_STANDALONE=OFF",
            "-DFL_TEXT_BUILD_TESTS=OFF",
            "-DFL_TEXT_BUILD_PYTHON=ON",
            "-DFL_TEXT_BUILD_PYTHON_PACKAGE=ON",
            "-DFL_TEXT_USE_KENLM=" + ("ON" if use_kenlm else "OFF"),
            "-DKENLM_LIB_PATH=" + str(kenlm_lib_path),
            "-DKENLM_HEADER_PATH=" + str(kenlm_header_path),
        ]
        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        if platform.system() == "Windows":
            cmake_args += [
                "-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), ext_dir),
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

        subprocess.check_call(
            ["cmake", source_dir] + cmake_args, cwd=self.build_temp, env=env
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )


def main():
    if os.getenv("BUILD_VERSION"):
        version = os.getenv("BUILD_VERSION")
    elif BUILD_VERSION_PATH.is_file():
        version = BUILD_VERSION_PATH.read_text().strip()
    elif VERSION_PY_PATH.is_file():
        # see if version.py is already written (as is the case with a source distribution)
        from bindings.python.flashlight.lib.text.version import __version__ as version
    elif VERSION_TXT_PATH.is_file():
        with open(VERSION_TXT_PATH) as f:
            version = f.readline().strip()
        version += get_local_version_suffix()
    else:
        raise RuntimeError(
            "Could not find version.txt, BUILD_VERSION.txt, or version.py. "
            "Please run `python bindings/python/compute_version.py` to "
            "generate the version file."
        )

    write_version_file(version)

    # Read Python bindings README
    long_description = (Path(PACKAGE_DIR) / "README.md").read_text()

    setup(
        name="flashlight-text",
        version=version,
        url="https://github.com/flashlight/text",
        author="Jacob Kahn",
        author_email="jacobkahn1@gmail.com",
        description="Flashlight Text bindings for Python",
        long_description=long_description,
        long_description_content_type="text/markdown",
        packages=find_namespace_packages(
            where=PACKAGE_DIR,
            include=["flashlight.lib.text", "flashlight.lib.text.decoder"],
            exclude=["test"],
        ),
        package_dir={"": PACKAGE_DIR},
        ext_modules=[
            CMakeExtension("flashlight.lib.text.decoder"),
            CMakeExtension("flashlight.lib.text.dictionary"),
        ],
        cmdclass={"build_ext": CMakeBuild},
        zip_safe=False,
        license="BSD licensed, as found in the LICENSE file",
        python_requires=">=3.6",
        classifiers=[
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Operating System :: OS Independent",
        ],
    )


if __name__ == "__main__":
    main()
