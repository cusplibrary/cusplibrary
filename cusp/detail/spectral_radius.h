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


#pragma once

#include <cusp/multiply.h>
#include <cusp/array1d.h>

#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>

#include <thrust/detail/integer_traits.h>

namespace cusp
{
namespace detail
{

// http://burtleburtle.net/bob/hash/integer.html
inline
__host__ __device__
unsigned int hash32(unsigned int a)
{
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a <<  5);
    a = (a + 0xd3a2646c) ^ (a <<  9);
    a = (a + 0xfd7046c5) + (a <<  3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

template <typename I, typename T>
struct hash_01
{
    __host__ __device__
    T operator()(const I& index) const
    {
        return T(hash32(index)) / T(thrust::detail::integer_traits<unsigned int>::const_max);
    }
};


template <typename Matrix>    
double estimate_spectral_radius(const Matrix& A, size_t k = 20)
{
    CUSP_PROFILE_SCOPED();

    typedef typename Matrix::index_type   IndexType;
    typedef typename Matrix::value_type   ValueType;
    typedef typename Matrix::memory_space MemorySpace;

    const IndexType N = A.num_rows;

    cusp::array1d<ValueType, MemorySpace> x(N);
    cusp::array1d<ValueType, MemorySpace> y(N);

    // initialize x to random values in [0,1)
    thrust::transform(thrust::counting_iterator<IndexType>(0),
                      thrust::counting_iterator<IndexType>(N),
                      x.begin(),
                      hash_01<IndexType,ValueType>());

    for(size_t i = 0; i < k; i++)
    {
        cusp::blas::scal(x, ValueType(1.0) / cusp::blas::nrmmax(x));
        cusp::multiply(A, x, y);
        x.swap(y);
    }
   
    if (k == 0)
        return 0;
    else
        return cusp::blas::nrm2(x) / cusp::blas::nrm2(y);
}

} // end namespace detail
} // end namespace cusp

