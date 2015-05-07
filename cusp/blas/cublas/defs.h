/*
 *  Copyright 2008-2009 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*! \file defs.h
 *  \brief CBLAS utility definitions for interface routines
 */

#pragma once

#include <cublas_v2.h>

namespace cusp
{
namespace blas
{
namespace cublas
{

struct cublas_format {};

struct upper   : public cublas_format {};
struct lower   : public cublas_format {};
struct unit    : public cublas_format {};
struct nonunit : public cublas_format {};

template< typename LayoutFormat >
struct Orientation {static const cublasOperation_t type;};
template<>
const cublasOperation_t Orientation<cusp::row_major>::type    = CUBLAS_OP_T;
template<>
const cublasOperation_t Orientation<cusp::column_major>::type = CUBLAS_OP_N;

} // end namespace cublas
} // end namespace blas
} // end namespace cusp

