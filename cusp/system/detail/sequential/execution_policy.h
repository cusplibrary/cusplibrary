/*
 *  Copyright 2008-2013 NVIDIA Corporation
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

#pragma once

/*! \file cusp/system/detail/sequential/execution_policy.h
 *  \brief Execution policies for Cusp's standard sequential system.
 */

#include <cusp/detail/config.h>

#if THRUST_VERSION >= 100800
// get the execution policies definitions first
#include <thrust/system/detail/sequential/execution_policy.h>
#else
#include <cusp/detail/thrust/system/detail/sequential/execution_policy.h>
#endif

namespace cusp
{
namespace system
{
namespace detail
{
namespace sequential
{
using namespace thrust::system::detail::sequential;
} // end namespace sequential
} // end namespace detail
} // end namespace system
} // end namespace cusp

// now get all the algorithm definitions

#include <cusp/system/detail/sequential/convert.h>
#include <cusp/system/detail/sequential/elementwise.h>
#include <cusp/system/detail/sequential/multiply.h>
#include <cusp/system/detail/sequential/transpose.h>

#include <cusp/system/detail/sequential/graph/breadth_first_search.h>
