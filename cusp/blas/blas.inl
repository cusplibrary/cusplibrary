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

#include <cusp/array1d.h>
#include <cusp/complex.h>
#include <cusp/exception.h>

#include <cusp/blas/blas_policy.h>

namespace cusp
{
namespace blas
{
namespace detail
{

template <typename Array1, typename Array2>
void assert_same_dimensions(const Array1& array1,
                            const Array2& array2)
{
    if(array1.size() != array2.size())
        throw cusp::invalid_input_exception("array dimensions do not match");
}

template <typename Array1, typename Array2, typename Array3>
void assert_same_dimensions(const Array1& array1,
                            const Array2& array2,
                            const Array3& array3)
{
    assert_same_dimensions(array1, array2);
    assert_same_dimensions(array2, array3);
}

template <typename Array1, typename Array2, typename Array3, typename Array4>
void assert_same_dimensions(const Array1& array1,
                            const Array2& array2,
                            const Array3& array3,
                            const Array4& array4)
{
    assert_same_dimensions(array1, array2);
    assert_same_dimensions(array2, array3);
    assert_same_dimensions(array3, array4);
}
} // end namespace detail

template <typename Array1,
         typename Array2,
         typename ScalarType>
void axpy(const blas_policy<typename Array2::value_type,typename Array2::memory_space>& policy,
          const Array1& x,
          Array2& y,
          ScalarType alpha)
{
    typedef typename Array2::value_type ValueType;
    typedef typename Array2::memory_space MemorySpace;
    typedef typename blas_policy<ValueType,MemorySpace>::type DerivedPolicy;

    CUSP_PROFILE_SCOPED();

    detail::assert_same_dimensions(x, y);
    cusp::blas::axpy(DerivedPolicy(), x, y, alpha);
}

template <typename Array1,
         typename Array2,
         typename ScalarType>
void axpy(const Array1& x,
          Array2& y,
          ScalarType alpha)
{
    typedef typename Array2::value_type ValueType;
    typedef typename Array2::memory_space MemorySpace;
    blas_policy<ValueType,MemorySpace> policy;
    cusp::blas::axpy(policy, x, y, alpha);
}

template <typename Array1,
         typename Array2,
         typename ScalarType>
void axpy(const Array1& x,
          const Array2& y,
          ScalarType alpha)
{
    cusp::blas::axpy(x, const_cast<Array2&>(y), alpha);
}

template <typename Array1,
         typename Array2,
         typename Array3,
         typename ScalarType1,
         typename ScalarType2>
void axpby(const blas_policy<typename Array3::value_type,typename Array3::memory_space>& policy,
           const Array1& x,
           const Array2& y,
           Array3& z,
           ScalarType1 alpha,
           ScalarType2 beta)
{
    typedef typename Array3::value_type ValueType;
    typedef typename Array3::memory_space MemorySpace;
    typedef typename blas_policy<ValueType,MemorySpace>::type DerivedPolicy;

    CUSP_PROFILE_SCOPED();

    detail::assert_same_dimensions(x, y, z);
    cusp::blas::axpby(DerivedPolicy(), x, y, z, alpha, beta);
}

template <typename Array1,
         typename Array2,
         typename Array3,
         typename ScalarType1,
         typename ScalarType2>
void axpby(const Array1& x,
           const Array2& y,
           Array3& z,
           ScalarType1 alpha,
           ScalarType2 beta)
{
    typedef typename Array3::value_type ValueType;
    typedef typename Array3::memory_space MemorySpace;
    blas_policy<ValueType,MemorySpace> policy;
    cusp::blas::axpby(policy, x, y, z, alpha, beta);
}

template <typename Array1,
         typename Array2,
         typename Array3,
         typename ScalarType1,
         typename ScalarType2>
void axpby(const Array1& x,
           const Array2& y,
           const Array3& z,
           ScalarType1 alpha,
           ScalarType2 beta)
{
    cusp::blas::axpby(x, y, const_cast<Array3&>(z), alpha, beta);
}

template <typename Array1,
         typename Array2,
         typename Array3,
         typename Array4,
         typename ScalarType1,
         typename ScalarType2,
         typename ScalarType3>
void axpbypcz(const blas_policy<typename Array4::value_type,typename Array4::memory_space>& policy,
              const Array1& x,
              const Array2& y,
              const Array3& z,
              Array4& output,
              ScalarType1 alpha,
              ScalarType2 beta,
              ScalarType3 gamma)
{
    typedef typename Array4::value_type ValueType;
    typedef typename Array4::memory_space MemorySpace;
    typedef typename blas_policy<ValueType,MemorySpace>::type DerivedPolicy;

    CUSP_PROFILE_SCOPED();

    detail::assert_same_dimensions(x, y, z, output);
    cusp::blas::axpbypcz(DerivedPolicy(), x, y, z, output, alpha, beta, gamma);
}

template <typename Array1,
         typename Array2,
         typename Array3,
         typename Array4,
         typename ScalarType1,
         typename ScalarType2,
         typename ScalarType3>
void axpbypcz(const Array1& x,
              const Array2& y,
              const Array3& z,
              Array4& output,
              ScalarType1 alpha,
              ScalarType2 beta,
              ScalarType3 gamma)
{
    typedef typename Array4::value_type ValueType;
    typedef typename Array4::memory_space MemorySpace;
    blas_policy<ValueType,MemorySpace> policy;
    cusp::blas::axpbypcz(policy, x, y, z, output, alpha, beta, gamma);
}

template <typename Array1,
         typename Array2,
         typename Array3,
         typename Array4,
         typename ScalarType1,
         typename ScalarType2,
         typename ScalarType3>
void axpbypcz(const Array1& x,
              const Array2& y,
              const Array3& z,
              const Array4& output,
              ScalarType1 alpha,
              ScalarType2 beta,
              ScalarType3 gamma)
{
    cusp::blas::axpbypcz(x, y, z, const_cast<Array4&>(output), alpha, beta, gamma);
}

template <typename Array1,
         typename Array2,
         typename Array3>
void xmy(const blas_policy<typename Array3::value_type,typename Array3::memory_space>& policy,
         const Array1& x,
         const Array2& y,
         Array3& output)
{
    typedef typename Array3::value_type ValueType;
    typedef typename Array3::memory_space MemorySpace;
    typedef typename blas_policy<ValueType,MemorySpace>::type DerivedPolicy;

    CUSP_PROFILE_SCOPED();

    detail::assert_same_dimensions(x, y, output);
    cusp::blas::xmy(DerivedPolicy(), x, y, output);
}

template <typename Array1,
         typename Array2,
         typename Array3>
void xmy(const Array1& x,
         const Array2& y,
         Array3& output)
{
    typedef typename Array1::value_type ValueType;
    typedef typename Array1::memory_space MemorySpace;
    blas_policy<ValueType,MemorySpace> policy;
    cusp::blas::xmy(policy, x, y, output);
}

template <typename Array1,
         typename Array2,
         typename Array3>
void xmy(const Array1& x,
         const Array2& y,
         const Array3& output)
{
    cusp::blas::xmy(x, y, const_cast<Array3&>(output));
}

template <typename Array1,
         typename Array2>
void copy(const Array1& x,
          Array2& y)
{
    CUSP_PROFILE_SCOPED();

    detail::assert_same_dimensions(x, y);
    thrust::copy(x.begin(), x.end(), y.begin());
}

template <typename Array1,
         typename Array2>
void copy(const Array1& x,
          const Array2& y)
{
    cusp::blas::copy(x, const_cast<Array2&>(y));
}

// TODO properly harmonize heterogenous types
template <typename Array1,
         typename Array2>
typename Array1::value_type
dot(const blas_policy<typename Array1::value_type,typename Array1::memory_space>& policy,
    const Array1& x,
    const Array2& y)
{
    typedef typename Array1::value_type ValueType;
    typedef typename Array1::memory_space MemorySpace;
    typedef typename blas_policy<ValueType,MemorySpace>::type DerivedPolicy;

    CUSP_PROFILE_SCOPED();

    return cusp::blas::dot(DerivedPolicy(), x, y);
}

template <typename Array1,
         typename Array2>
typename Array1::value_type
dot(const Array1& x,
    const Array2& y)
{
    detail::assert_same_dimensions(x, y);

    typedef typename Array1::value_type ValueType;
    typedef typename Array1::memory_space MemorySpace;
    blas_policy<ValueType,MemorySpace> policy;
    return cusp::blas::dot(policy, x, y);
}

// TODO properly harmonize heterogenous types
template <typename Array1,
         typename Array2>
typename Array1::value_type
dotc(const blas_policy<typename Array1::value_type,typename Array1::memory_space>& policy,
     const Array1& x,
     const Array2& y)
{
    typedef typename Array1::value_type ValueType;
    typedef typename Array1::memory_space MemorySpace;
    typedef typename blas_policy<ValueType,MemorySpace>::type DerivedPolicy;

    CUSP_PROFILE_SCOPED();

    return cusp::blas::dotc(DerivedPolicy(), x, y);
}

template <typename Array1,
         typename Array2>
typename Array1::value_type
dotc(const Array1& x,
     const Array2& y)
{
    detail::assert_same_dimensions(x, y);

    typedef typename Array1::value_type ValueType;
    typedef typename Array1::memory_space MemorySpace;
    blas_policy<ValueType,MemorySpace> policy;
    return cusp::blas::dotc(policy, x, y);
}

template <typename Array,
         typename ScalarType>
void fill(Array& x,
          ScalarType alpha)
{
    thrust::fill(x.begin(), x.end(), alpha);
}

template <typename Array,
         typename ScalarType>
void fill(const Array& x,
          ScalarType alpha)
{
    cusp::blas::fill(const_cast<Array&>(x), alpha);
}

template <typename Array>
typename norm_type<typename Array::value_type>::type
nrm1(const blas_policy<typename Array::value_type,typename Array::memory_space>& policy,
     const Array& x)
{
    typedef typename Array::value_type ValueType;
    typedef typename Array::memory_space MemorySpace;
    typedef typename blas_policy<ValueType,MemorySpace>::type DerivedPolicy;

    CUSP_PROFILE_SCOPED();

    return cusp::blas::nrm1(DerivedPolicy(), x);
}

template <typename Array>
typename norm_type<typename Array::value_type>::type
nrm1(const Array& x)
{
    typedef typename Array::value_type ValueType;
    typedef typename Array::memory_space MemorySpace;
    blas_policy<ValueType,MemorySpace> policy;
    return cusp::blas::nrm1(policy, x);
}

template <typename Array>
typename norm_type<typename Array::value_type>::type
nrm2(const blas_policy<typename Array::value_type,typename Array::memory_space>& policy,
     const Array& x)
{
    typedef typename Array::value_type ValueType;
    typedef typename Array::memory_space MemorySpace;
    typedef typename blas_policy<ValueType,MemorySpace>::type DerivedPolicy;

    CUSP_PROFILE_SCOPED();

    return cusp::blas::nrm2(DerivedPolicy(), x);
}

template <typename Array>
typename norm_type<typename Array::value_type>::type
nrm2(const Array& x)
{
    typedef typename Array::value_type ValueType;
    typedef typename Array::memory_space MemorySpace;
    blas_policy<ValueType,MemorySpace> policy;

    return cusp::blas::nrm2(policy, x);
}

template <typename Array>
typename Array::value_type
nrmmax(const blas_policy<typename Array::value_type,typename Array::memory_space>& policy,
       const Array& x)
{
    typedef typename Array::value_type ValueType;
    typedef typename Array::memory_space MemorySpace;
    typedef typename blas_policy<ValueType,MemorySpace>::type DerivedPolicy;

    CUSP_PROFILE_SCOPED();

    return cusp::blas::nrmmax(DerivedPolicy(), x);
}

template <typename Array>
typename Array::value_type
nrmmax(const Array& x)
{
    typedef typename Array::value_type ValueType;
    typedef typename Array::memory_space MemorySpace;
    blas_policy<ValueType,MemorySpace> policy;
    return cusp::blas::nrmmax(policy, x);
}

template <typename Array,
         typename ScalarType>
void scal(const blas_policy<typename Array::value_type,typename Array::memory_space>& policy,
          Array& x,
          ScalarType alpha)
{
    typedef typename Array::value_type ValueType;
    typedef typename Array::memory_space MemorySpace;
    typedef typename blas_policy<ValueType,MemorySpace>::type DerivedPolicy;

    CUSP_PROFILE_SCOPED();

    cusp::blas::scal(DerivedPolicy(), x, alpha);
}

template <typename Array,
         typename ScalarType>
void scal(Array& x,
          ScalarType alpha)
{
    typedef typename Array::value_type ValueType;
    typedef typename Array::memory_space MemorySpace;
    blas_policy<ValueType,MemorySpace> policy;
    cusp::blas::scal(policy, x, alpha);
}

template <typename Array,
         typename ScalarType>
void scal(const Array& x,
          ScalarType alpha)
{
    cusp::blas::scal(const_cast<Array&>(x), alpha);
}
} // end namespace blas
} // end namespace cusp

