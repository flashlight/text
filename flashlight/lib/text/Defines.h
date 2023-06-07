/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#if defined(_WIN32) || defined(_MSC_VER)

#ifdef FL_TEXT_DLL
#define FL_TEXT_API __declspec(dllexport)
#else // FL_TEXT_DLL
#define FL_TEXT_API __declspec(dllimport)
#endif // FL_TEXT_DLL

#else // defined(_WIN32) || defined(_MSC_VER)

#define FL_TEXT_API __attribute__((visibility("default")))
#define FL_DEPRECATED(msg) __attribute__((deprecated(msg)))

#endif // defined(_WIN32) || defined(_MSC_VER)
