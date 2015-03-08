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

/*! \file thrust/system/tbb/execution_policy.h
 *  \brief Execution policies for Thrust's OpenMP system.
 */

#include <thrust/detail/config.h>

// get the execution policies definitions first
#include <thrust/system/omp/detail/execution_policy.h>

// get the definition of par
#include <thrust/system/omp/detail/par.h>

namespace cusp
{
namespace system
{
namespace omp
{
using namespace thrust::system::omp;
} // end namespace omp
} // end namespace system
} // end namespace cusp

// now get all the algorithm definitions

#include <cusp/system/omp/detail/convert.h>
#include <cusp/system/omp/detail/elementwise.h>
#include <cusp/system/omp/detail/multiply.h>
#include <cusp/system/omp/detail/sort.h>
#include <cusp/system/omp/detail/transpose.h>

#include <cusp/system/omp/detail/graph/breadth_first_search.h>
#include <cusp/system/omp/detail/graph/connected_components.h>
#include <cusp/system/omp/detail/graph/hilbert_curve.h>
#include <cusp/system/omp/detail/graph/maximal_independent_set.h>
#include <cusp/system/omp/detail/graph/pseudo_peripheral.h>
#include <cusp/system/omp/detail/graph/symmetric_rcm.h>
#include <cusp/system/omp/detail/graph/vertex_coloring.h>
