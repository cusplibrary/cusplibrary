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

#include <cusp/exception.h>
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>

#include <cusp/graph/detail/dispatch/maximal_independent_set.h>

#include <thrust/fill.h>

namespace cusp
{
namespace graph
{
namespace detail
{

//////////////////
// General Path //
//////////////////
template <typename Matrix, typename Array>
size_t maximal_independent_set(const Matrix& A, Array& stencil, size_t k,
                               cusp::coo_format, cusp::device_memory)
{
  typedef typename Matrix::index_type   IndexType;
  typedef typename Matrix::value_type   ValueType;

  return cusp::graph::detail::dispatch::maximal_independent_set(A, stencil, k, typename Matrix::memory_space());
}

template <typename Matrix, typename Array,
          typename Format>
size_t maximal_independent_set(const Matrix& A, Array& stencil, size_t k,
                               Format, cusp::device_memory)
{
  typedef typename Matrix::index_type   IndexType;
  typedef typename Matrix::value_type   ValueType;

  // convert matrix to COO format and compute on the device
  cusp::coo_matrix<IndexType,ValueType,cusp::device_memory> A_coo(A);

  return cusp::graph::detail::dispatch::maximal_independent_set(A_coo, stencil, k, typename Matrix::memory_space());
}

template <typename Matrix, typename Array>
size_t maximal_independent_set(const Matrix& A, Array& stencil, size_t k,
                               cusp::csr_format, cusp::host_memory)
{
  typedef typename Matrix::index_type   IndexType;
  typedef typename Matrix::value_type   ValueType;

  return cusp::graph::detail::dispatch::maximal_independent_set(A, stencil, k, typename Matrix::memory_space());
}

template <typename Matrix, typename Array,
          typename Format>
size_t maximal_independent_set(const Matrix& A, Array& stencil, size_t k,
                               Format, cusp::host_memory)
{
  typedef typename Matrix::index_type   IndexType;
  typedef typename Matrix::value_type   ValueType;

  // convert matrix to CSR format and compute on the host
  cusp::csr_matrix<IndexType,ValueType,cusp::host_memory> A_csr(A);

  return cusp::graph::detail::dispatch::maximal_independent_set(A_csr, stencil, k, typename Matrix::memory_space());
}

} // end namespace detail

/////////////////
// Entry Point //
/////////////////

template <typename Matrix, typename Array>
size_t maximal_independent_set(const Matrix& A, Array& stencil, size_t k)
{
    CUSP_PROFILE_SCOPED();

    if(A.num_rows != A.num_cols)
        throw cusp::invalid_input_exception("matrix must be square");

    if (k == 0)
    {
        stencil.resize(A.num_rows);
        thrust::fill(stencil.begin(), stencil.end(), typename Array::value_type(1));
        return stencil.size();
    }
    else
    {
        return cusp::graph::detail::maximal_independent_set(A, stencil, k, typename Matrix::format(), typename Matrix::memory_space());
    }
}

} // end namespace graph
} // end namespace cusp

