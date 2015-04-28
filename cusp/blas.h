/*
 *  Copyright 2008-2014 NVIDIA Corporation
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

/*! \file blas.h
 *  \brief BLAS-like functions
 */

// TODO : Remove cusp/blas.h in v0.6.0
#pragma once

#include <cusp/detail/config.h>
#include <cusp/blas/blas.h>
#include <cusp/verify.h>

#if THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC
#    pragma message("| WARNING: cusp/blas.h is deprecated as of 0.4.0; use cusp/blas/blas.h instead |")
#else
#    warning | WARNING: cusp/blas.h is deprecated as of 0.4.0; use cusp/blas/blas.h instead |
#endif

namespace cusp
{
namespace blas
{
namespace detail
{
  using cusp::assert_same_dimensions;
  using cusp::is_valid_matrix;
} // end namespace detail
} // end namespace blas
} // end namespace cusp
