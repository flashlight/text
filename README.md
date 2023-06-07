# Flashlight Text: Fast, Lightweight Utilities for Text

[**Quickstart**](#quickstart)
| [**Installation**](#building-and-installing)
| [**Python Documentation**](bindings/python)
| [**Citing**](#citing)

[![CircleCI](https://circleci.com/gh/flashlight/text.svg?style=shield)](https://app.circleci.com/pipelines/github/flashlight/text) [![Join the chat at https://gitter.im/flashlight-ml/community](https://img.shields.io/gitter/room/flashlight-ml/community)](https://gitter.im/flashlight-ml/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) [![PyPI](https://img.shields.io/pypi/v/flashlight-text?color=dark%20green)](https://pypi.org/project/flashlight-text/) [![PyPI - Format](https://img.shields.io/pypi/format/flashlight-text)](https://pypi.org/project/flashlight-text/#files) [![vcpkg](https://img.shields.io/vcpkg/v/flashlight-text)](https://vcpkg.link/ports/flashlight-text) [![Codecov](https://img.shields.io/codecov/c/github/flashlight/text)](https://codecov.io/gh/flashlight/text) [![GitHub](https://img.shields.io/github/license/flashlight/text?color=light%20green)](https://github.com/flashlight/text/blob/main/LICENSE)

*Flashlight Text* is a fast, minimal library for text-based operations. It features:
- a high-performance, unopinionated [beam search decoder](flashlight/lib/text/decoder)
- a fast [tokenizer](flashlight/lib/text/tokenizer)
- an efficient [`Dictionary`](flashlight/lib/text/dictionary) abstraction

## Quickstart

The Flashlight Text Python package containing beam search decoder and Dictionary components is available on PyPI:
```bash
pip install flashlight-text
```
To enable optional KenLM support in Python with the decoder, KenLM must be installed via pip:
```bash
pip install git+https://github.com/kpu/kenlm.git
```

See the [full Python binding documentation](bindings/python) for examples and more.

## Building and Installing
[**From Source (C++)**](#building-from-source) | [**With `vcpkg` (C++)**](#with-vcpkg) | [**From Source (Python)**](bindings/python#build-instructions) | [**Adding to Your Own Project (C++)**](#adding-flashlight-text-to-a-c-project)

### Requirements
At minimum, C++ compilation requires:
- A C++ compiler with good C++17 support (e.g. gcc/g++ >= 7)
- [CMake](https://cmake.org/) — version 3.16 or later, and ``make``
- A Linux-based operating system.

**KenLM Support:** If building with KenLM support, [KenLM](https://github.com/kpu/kenlm/) is required. To toggle KenLM support use the `FL_TEXT_USE_KENLM` CMake option or the `USE_KENLM` environment variable when building the Python bindings.

**Tests:** If building tests, [Google Test](https://github.com/google/googletest) >= 1.10 is required. The `FL_TEXT_BUILD_TESTS` CMake option toggles building tests.

Instructions for building/installing the Python bindings from source [can be found here](bindings/python/README.md).

### Building from Source

Building the C++ project from source is simple:
```bash
git clone https://github.com/flashlight/text && cd text
cmake -S . -B build
cmake --build build --parallel
cd build && ctest && cd .. # run tests
cmake --install build # install at the CMAKE_INSTALL_PREFIX
```
To disable KenLM while building, pass `-DFL_TEXT_USE_KENLM=OFF` to CMake. To disable building tests, pass `-DFL_TEXT_BUILD_TESTS=OFF`.

KenLM can be downloaded and installed automatically if not found on the local system. The `FL_TEXT_BUILD_STANDALONE` option controls this behavior — if disabled, dependencies won't be downloaded and built when building.

#### With [`vcpkg`](https://vcpkg.io/)

Flashlight Text can also be installed and used downstream with the [`vcpkg`](https://vcpkg.io/) package manager. The [port](https://github.com/microsoft/vcpkg/blob/master/ports/flashlight-text/) contains an optional feature with which to build and install with KenLM support:
```bash
vcpkg install flashlight-text # no dependencies, or:
vcpkg install "flashlight-text[kenlm]" # install with KenLM
```

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

To link against the library providing KenLM support, use the `flashlight::flashlight-text-kenlm` imported target:
```cmake
target_link_libraries(myProject
  PRIVATE
  flashlight::flashlight-text
  # transitively links KenLM
  flashlight::flashlight-text-kenlm
)
```

### Contributing and Contact
Contact: jacobkahn@meta.com

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
