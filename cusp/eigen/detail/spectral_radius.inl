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


#pragma once

#include <cusp/array1d.h>
#include <cusp/array2d.h>
#include <cusp/format_utils.h>
#include <cusp/multiply.h>

#include <cusp/blas/blas.h>
#include <cusp/eigen/arnoldi.h>

#include <thrust/extrema.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>

#include <thrust/detail/integer_traits.h>

#include <algorithm>

namespace cusp
{
namespace eigen
{
namespace detail
{

template <typename Matrix>
double disks_spectral_radius(const Matrix& A, coo_format)
{
    typedef typename Matrix::index_type   IndexType;
    typedef typename Matrix::value_type   ValueType;
    typedef typename Matrix::memory_space MemorySpace;

    const IndexType N = A.num_rows;

    // compute sum of absolute values for each row of A
    cusp::array1d<IndexType, MemorySpace> row_sums(N);

    cusp::array1d<IndexType, MemorySpace> temp(N);
    thrust::reduce_by_key
    (A.row_indices.begin(), A.row_indices.end(),
     thrust::make_transform_iterator(A.values.begin(), cusp::detail::absolute<ValueType>()),
     temp.begin(),
     row_sums.begin());

    return *thrust::max_element(row_sums.begin(), row_sums.end());
}

template <typename Matrix>
double disks_spectral_radius(const Matrix& A, sparse_format)
{
    typename cusp::detail::as_coo_type<Matrix>::type A_coo(A);
    return disks_spectral_radius(A_coo, coo_format());
}

template <typename Matrix, typename Array2d>
void lanczos_estimate(const Matrix& A, Array2d& H, size_t k)
{
    typedef typename Matrix::value_type   ValueType;
    typedef typename Matrix::memory_space MemorySpace;

    size_t N = A.num_cols;
    size_t maxiter = std::min(N, k);

    // allocate workspace
    cusp::array1d<ValueType,MemorySpace> v0(N);
    cusp::array1d<ValueType,MemorySpace> v1(N);
    cusp::array1d<ValueType,MemorySpace> w(N);

    // initialize starting vector to random values in [0,1)
    cusp::copy(cusp::random_array<ValueType>(N), v1);

    cusp::blas::scal(v1, ValueType(1) / cusp::blas::nrm2(v1));

    Array2d H_(maxiter + 1, maxiter, 0);

    ValueType alpha = 0.0, beta = 0.0;

    size_t j;

    for(j = 0; j < maxiter; j++)
    {
        cusp::multiply(A, v1, w);

        if(j >= 1)
        {
            H_(j - 1, j) = beta;
            cusp::blas::axpy(v0, w, -beta);
        }

        alpha = cusp::blas::dot(w, v1);
        H_(j,j) = alpha;

        cusp::blas::axpy(v1, w, -alpha);

        beta = cusp::blas::nrm2(w);
        H_(j + 1, j) = beta;

        if(beta < 1e-10) break;

        cusp::blas::scal(w, ValueType(1) / beta);

        // [v0 v1  w] - > [v1  w v0]
        v0.swap(v1);
        v1.swap(w);
    }

    H.resize(j,j);
    for(size_t row = 0; row < j; row++)
        for(size_t col = 0; col < j; col++)
            H(row,col) = H_(row,col);
}

} // end detail namespace

template <typename Matrix>
double estimate_spectral_radius(const Matrix& A, size_t k)
{
    typedef typename Matrix::index_type   IndexType;
    typedef typename Matrix::value_type   ValueType;
    typedef typename Matrix::memory_space MemorySpace;

    const IndexType N = A.num_rows;

    cusp::array1d<ValueType, MemorySpace> x(N);
    cusp::array1d<ValueType, MemorySpace> y(N);

    // initialize x to random values in [0,1)
    cusp::copy(cusp::random_array<ValueType>(N), x);

    for(size_t i = 0; i < k; i++)
    {
        cusp::blas::scal(x, ValueType(1.0) / cusp::blas::nrmmax(x));
        cusp::multiply(A, x, y);
        x.swap(y);
    }

    return k == 0 ? 0 : cusp::blas::nrm2(x) / cusp::blas::nrm2(y);
}

template <typename Matrix>
double ritz_spectral_radius(const Matrix& A, size_t k, bool symmetric)
{
    typedef typename Matrix::value_type ValueType;

    cusp::array2d<ValueType,cusp::host_memory> H;

    if(symmetric)
      detail::lanczos_estimate(A, H, k);
    else
      cusp::eigen::arnoldi(A, H, k);

    return estimate_spectral_radius(H);
}

template <typename Matrix>
double disks_spectral_radius(const Matrix& A)
{
    return detail::disks_spectral_radius(A, typename Matrix::format());
}

} // end namespace eigen
} // end namespace cusp

