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

/*! \file jacobi.inl
 *  \brief Inline file for jacobi.h
 */

#include <cusp/multiply.h>
#include <cusp/detail/format_utils.h>

#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>

namespace cusp
{
namespace relaxation
{
namespace detail
{

template <typename T>
struct jacobi_functor
{
    template <typename Tuple>
    T operator()(const Tuple& t)
    {
        const T x = thrust::get<0>(t);
        const T d = thrust::get<1>(t);
        const T b = thrust::get<2>(t);
        const T y = thrust::get<3>(t);

        return x + (b - y) / d;
    }
};

} // end namespace detail


// constructor
template <typename ValueType, typename MemorySpace>
template<typename MatrixType>
    jacobi<ValueType,MemorySpace>
    ::jacobi(const MatrixType& A)
    {
        // extract the main diagonal
        cusp::detail::extract_diagonal(A, diagonal);
    }
        
// linear operator
template <typename ValueType, typename MemorySpace>
template<typename MatrixType, typename VectorType1, typename VectorType2>
    void jacobi<ValueType,MemorySpace>
    ::operator()(const MatrixType& A, const VectorType1& b, VectorType2& x)
    {
        // TODO see if preallocating y is noticably faster
        cusp::array1d<ValueType,MemorySpace> y(x.size());

        // y <- A*x
        cusp::multiply(A, x, y);
        
        // x <- D^-1 (b - y)
        thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(x.begin(), diagonal.begin(), b.begin(), y.begin())),
                          thrust::make_zip_iterator(thrust::make_tuple(x.end(),   diagonal.end(),   b.end(),   y.end())),
                          x.begin(),
                          detail::jacobi_functor<ValueType>());
    }

} // end namespace relaxation
} // end namespace cusp

