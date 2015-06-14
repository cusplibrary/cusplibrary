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

#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>

#include <cusp/multiply.h>
#include <cusp/graph/maximal_independent_set.h>

#include <thrust/count.h>
#include <thrust/fill.h>
#include <thrust/scan.h>
#include <thrust/gather.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

namespace cusp
{
namespace precond
{
namespace aggregation
{
namespace detail
{

template <typename DerivedPolicy, typename MatrixType, typename ArrayType>
void standard_aggregate(thrust::system::detail::sequential::execution_policy<DerivedPolicy> &exec,
                        const MatrixType& A, ArrayType& aggregates, ArrayType& roots,
                        cusp::known_format)
{
}

template <typename DerivedPolicy, typename MatrixType, typename ArrayType>
void standard_aggregate(thrust::system::detail::sequential::execution_policy<DerivedPolicy> &exec,
                        const MatrixType& A, ArrayType& aggregates, ArrayType& roots)
{
    typedef typename MatrixType::format Format;

    Format format;

    standard_aggregate(exec, A, aggregates, roots, format);
}

} // end namespace detail
} // end namespace aggregation
} // end namespace precond
} // end namespace cusp

