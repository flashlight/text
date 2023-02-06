# Flashlight Text: Fast, Lightweight Utilities for Text

[**Quickstart**](#quickstart)
| [**Installation**](#building-and-installing)
| [**Python Documentation**](bindings/python)
| [**Citing**](#citing)

[![CircleCI](https://circleci.com/gh/flashlight/text.svg?style=shield)](https://app.circleci.com/pipelines/github/flashlight/text)
[![Join the chat at https://gitter.im/flashlight-ml/community](https://img.shields.io/gitter/room/flashlight-ml/community)](https://gitter.im/flashlight-ml/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) [![codecov](https://codecov.io/gh/flashlight/text/branch/main/graph/badge.svg?token=rBp4AilMc0)](https://codecov.io/gh/flashlight/text)

*Flashlight Text* is a fast, minimal library for text-based operations. It features:
- a high-performance, unopinionated [beam search decoder](flashlight/lib/text/decoder)
- a fast [tokenizer](flashlight/lib/text/tokenizer)
- an efficient [`Dictionary`](flashlight/lib/text/dictionary) abstraction

## Quickstart

The Flashlight Text Python package containing decoder and Dictionary components is available on PyPI **without KenLM support** via:
```bash
pip install flashlight-text  # without KenLM support
```
For now, building from source is required for KenLM support. We'll be adding KenLM support to the PyPI package soon.

To install the bindings from source, [install KenLM](https://github.com/kpu/kenlm/), then clone the repo and build:
```shell
git clone https://github.com/flashlight/text && cd text
pip install .
```
To install without KenLM when building from source, set the environment variable `USE_KENLM=0` when running `pip install .`.

See the [full Python binding documentation](bindings/python) for examples and more.

## Building and Installing
[**From Source (C++)**](#building-from-source) | [**From Source (Python)**](bindings/python#build-instructions) | [**Adding to Your Own Project (C++)**](#adding-flashlight-text-to-a-c++-project)

### Requirements
At minimum, compilation requires:
- A C++ compiler with good C++17 support (e.g. gcc/g++ >= 7)
- [CMake](https://cmake.org/) — version 3.10 or later, and ``make``
- A Linux-based operating system.

**KenLM Support:** If building with KenLM support, [KenLM](https://github.com/kpu/kenlm/) is required. To toggle KenLM support use the `FL_TEXT_USE_KENLM` CMake option or the `USE_KENLM` environment variable when building the Python bindings.

**Tests:** If building tests, [Google Test](https://github.com/google/googletest) >= 1.10 is required. The `FL_TEXT_BUILD_TESTS` CMake option toggles building tests.

Instructions for building/installing the Python bindings from source [can be found here](bindings/python/README.md).

### Building from Source

Building the C++ project from source is simple:
```bash
git clone https://github.com/flashlight/text && cd flashlight
mkdir build && cd build
cmake ..
make -j$(nproc)
make test    # run tests
make install # install at the CMAKE_INSTALL_PREFIX
```
To disable KenLM while building, pass `-DFL_TEXT_USE_KENLM=OFF` to CMake. To disable building tests, pass `-DFL_TEXT_BUILD_TESTS=OFF`.

KenLM can be downloaded and installed automatically if not found on the local system. The `FL_TEXT_BUILD_STANDALONE` option controls this behavior — if disabled, dependencies won't be downloaded and built when building.

### Adding Flashlight Text to a C++ Project

Given a simple `project.cpp` file that includes and links to Flashlight Text:
```c++
#include <iostream>

#include <flashlight/lib/text/dictionary/Dictionary.h>

int main() {
  fl::lib::text::Dictionary myDict("someFile.dict");
  std::cout << "Dictionary has " << myDict.entrySize()
            << " entries."  << std::endl;
 return 0;
}
```

The following CMake configuration links Flashlight and sets include directories:

```cmake
cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(myProject project.cpp)

find_package(flashlight-text CONFIG REQUIRED)
target_link_libraries(myProject PRIVATE flashlight::flashlight-text)
```

### Contributing and Contact
Contact: jacobkahn@fb.com

Flashlight Text is actively developed. See
[CONTRIBUTING](CONTRIBUTING.md) for more on how to help out.

## Citing
You can cite [Flashlight](https://arxiv.org/abs/2201.12465) using:
```
@misc{kahn2022flashlight,
      title={Flashlight: Enabling Innovation in Tools for Machine Learning},
      author={Jacob Kahn and Vineel Pratap and Tatiana Likhomanenko and Qiantong Xu and Awni Hannun and Jeff Cai and Paden Tomasello and Ann Lee and Edouard Grave and Gilad Avidov and Benoit Steiner and Vitaliy Liptchinsky and Gabriel Synnaeve and Ronan Collobert},
      year={2022},
      eprint={2201.12465},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## License
Flashlight Text is under an MIT license. See [LICENSE](LICENSE) for more information.
