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

#include <cusp/array1d.h>
#include <cusp/complex.h>
#include <cusp/exception.h>
#include <cusp/verify.h>

#include <cusp/execution_policy.h>
#include <cusp/blas/thrustblas/blas.h>

#include <thrust/iterator/iterator_traits.h>

namespace cusp
{
namespace blas
{

template <typename DerivedPolicy,
          typename Array1,
          typename Array2,
          typename ScalarType>
void axpy(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
          const Array1& x,
                Array2& y,
          const ScalarType alpha)
{
    using cusp::blas::thrustblas::axpy;

    cusp::assert_same_dimensions(x, y);

    axpy(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), x, y, alpha);
}

template <typename Array1,
          typename Array2,
          typename ScalarType>
void axpy(const Array1& x,
                Array2& y,
          const ScalarType alpha)
{
    using thrust::system::detail::generic::select_system;

    typedef typename Array1::memory_space System1;
    typedef typename Array2::memory_space System2;

    System1 system1;
    System2 system2;

    cusp::blas::axpy(select_system(system1,system2), x, y, alpha);
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
    using thrust::system::detail::generic::select_system;

    typedef typename Array1::memory_space System1;
    typedef typename Array2::memory_space System2;
    typedef typename Array3::memory_space System3;

    System1 system1;
    System2 system2;
    System3 system3;

    cusp::blas::axpby(select_system(system1,system2,system3), x, y, z, alpha, beta);
}

template <typename DerivedPolicy,
          typename Array1,
          typename Array2,
          typename Array3,
          typename ScalarType1,
          typename ScalarType2>
void axpby(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
           const Array1& x,
           const Array2& y,
                 Array3& z,
                 ScalarType1 alpha,
                 ScalarType2 beta)
{
    cusp::assert_same_dimensions(x, y, z);

    size_t N = x.size();

    thrust::for_each(thrust::detail::derived_cast(thrust::detail::strip_const(exec)),
                     thrust::make_zip_iterator(thrust::make_tuple(x.begin(), y.begin(), z.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(x.begin(), y.begin(), z.begin())) + N,
                     cusp::detail::AXPBY<ScalarType1,ScalarType2>(alpha, beta));
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
    using thrust::system::detail::generic::select_system;

    typedef typename Array1::memory_space System1;
    typedef typename Array2::memory_space System2;
    typedef typename Array3::memory_space System3;
    typedef typename Array4::memory_space System4;

    System1 system1;
    System2 system2;
    System3 system3;
    System4 system4;

    cusp::blas::axpbypcz(select_system(system1,system2,system3,system4), x, y, z, output, alpha, beta, gamma);
}

template <typename DerivedPolicy,
          typename Array1,
          typename Array2,
          typename Array3,
          typename Array4,
          typename ScalarType1,
          typename ScalarType2,
          typename ScalarType3>
void axpbypcz(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
              const Array1& x,
              const Array2& y,
              const Array3& z,
                    Array4& output,
                    ScalarType1 alpha,
                    ScalarType2 beta,
                    ScalarType3 gamma)
{
    cusp::assert_same_dimensions(x, y, z, output);

    size_t N = x.size();

    thrust::for_each(thrust::detail::derived_cast(thrust::detail::strip_const(exec)),
                     thrust::make_zip_iterator(thrust::make_tuple(x.begin(), y.begin(), z.begin(), output.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(x.begin(), y.begin(), z.begin(), output.begin())) + N,
                     cusp::detail::AXPBYPCZ<ScalarType1,ScalarType2,ScalarType3>(alpha, beta, gamma));
}

template <typename Array1,
          typename Array2,
          typename Array3>
void xmy(const Array1& x,
         const Array2& y,
               Array3& z)
{
    using thrust::system::detail::generic::select_system;

    typedef typename Array1::memory_space System1;
    typedef typename Array2::memory_space System2;
    typedef typename Array3::memory_space System3;

    System1 system1;
    System2 system2;
    System3 system3;

    cusp::blas::xmy(select_system(system1,system2,system3), x, y, z);
}

template <typename DerivedPolicy,
          typename Array1,
          typename Array2,
          typename Array3>
void xmy(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
         const Array1& x,
         const Array2& y,
               Array3& z)
{
    typedef typename Array3::value_type ValueType;

    cusp::assert_same_dimensions(x, y, z);

    thrust::transform(thrust::detail::derived_cast(thrust::detail::strip_const(exec)),
                      x.begin(), x.end(), y.begin(), z.begin(), cusp::detail::XMY<ValueType>());
}

template <typename Array1,
          typename Array2>
void copy(const Array1& x,
                Array2& y)
{
    using thrust::system::detail::generic::select_system;

    typedef typename Array1::memory_space System1;
    typedef typename Array2::memory_space System2;

    System1 system1;
    System2 system2;

    cusp::blas::copy(select_system(system1,system2), x, y);
}

template <typename Array,
          typename RandomAccessIterator>
void copy(const Array& x,
          cusp::array1d_view<RandomAccessIterator> y)
{
    using thrust::system::detail::generic::select_system;

    typedef typename Array::memory_space System1;
    typedef typename thrust::iterator_system<RandomAccessIterator>::type System2;

    System1 system1;
    System2 system2;

    cusp::blas::copy(select_system(system1,system2), x, y);
}

template <typename DerivedPolicy,
          typename Array1,
          typename Array2>
void copy(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
          const Array1& x,
                Array2& y)
{
    using cusp::blas::thrustblas::copy;

    cusp::assert_same_dimensions(x, y);

    copy(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), x, y);
}

template <typename Array1,
          typename Array2>
typename Array1::value_type
dot(const Array1& x,
    const Array2& y)
{
    using thrust::system::detail::generic::select_system;

    typedef typename Array1::memory_space System1;
    typedef typename Array2::memory_space System2;

    System1 system1;
    System2 system2;

    return cusp::blas::dot(select_system(system1,system2), x, y);
}

template <typename DerivedPolicy,
          typename Array1,
          typename Array2>
typename Array1::value_type
dot(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
    const Array1& x,
    const Array2& y)
{
    using cusp::blas::thrustblas::dot;

    cusp::assert_same_dimensions(x, y);

    return dot(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), x, y);
}

// TODO properly harmonize heterogenous types
template <typename Array1,
          typename Array2>
typename Array1::value_type
dotc(const Array1& x,
     const Array2& y)
{
    using thrust::system::detail::generic::select_system;

    typedef typename Array1::memory_space System1;
    typedef typename Array2::memory_space System2;

    System1 system1;
    System2 system2;

    return cusp::blas::dotc(select_system(system1,system2), x, y);
}

template <typename DerivedPolicy,
          typename Array1,
          typename Array2>
typename Array1::value_type
dotc(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
     const Array1& x,
     const Array2& y)
{
    using cusp::blas::thrustblas::dotc;

    cusp::assert_same_dimensions(x, y);

    return dotc(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), x, y);
}

template <typename RandomAccessIterator,
          typename ScalarType>
void fill(cusp::array1d_view<RandomAccessIterator> x,
          const ScalarType alpha)
{
    using thrust::system::detail::generic::select_system;

    typedef typename thrust::iterator_system<RandomAccessIterator>::type System;

    System system;

    cusp::blas::fill(select_system(system), x, alpha);
}

template <typename Array,
          typename ScalarType>
void fill(Array& x,
          const ScalarType alpha)
{
    using thrust::system::detail::generic::select_system;

    typedef typename Array::memory_space System;

    System system;

    cusp::blas::fill(select_system(system), x, alpha);
}

template <typename DerivedPolicy,
          typename Array,
          typename ScalarType>
void fill(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
          Array& x,
          ScalarType alpha)
{
    using cusp::blas::thrustblas::fill;

    fill(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), x.begin(), x.end(), alpha);
}

template <typename Array>
typename cusp::detail::norm_type<typename Array::value_type>::type
nrm1(const Array& x)
{
    using thrust::system::detail::generic::select_system;

    typedef typename Array::memory_space System;

    System system;

    return cusp::blas::nrm1(select_system(system), x);
}

template <typename DerivedPolicy,
          typename Array>
typename cusp::detail::norm_type<typename Array::value_type>::type
nrm1(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
     const Array& x)
{
    using cusp::blas::thrustblas::nrm1;

    return nrm1(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), x);
}

template <typename Array>
typename cusp::detail::norm_type<typename Array::value_type>::type
nrm2(const Array& x)
{
    using thrust::system::detail::generic::select_system;

    typedef typename Array::memory_space System;

    System system;

    return cusp::blas::nrm2(select_system(system), x);
}

template <typename DerivedPolicy,
          typename Array>
typename cusp::detail::norm_type<typename Array::value_type>::type
nrm2(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
     const Array& x)
{
    using cusp::blas::thrustblas::nrm2;

    return nrm2(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), x);
}

template <typename Array>
typename Array::value_type
nrmmax(const Array& x)
{
    using thrust::system::detail::generic::select_system;

    typedef typename Array::memory_space System;

    System system;

    return cusp::blas::nrmmax(select_system(system), x);
}

template <typename DerivedPolicy,
          typename Array>
typename Array::value_type
nrmmax(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
       const Array& x)
{
    using cusp::blas::thrustblas::nrmmax;

    return nrmmax(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), x);
}

template <typename Array,
          typename ScalarType>
void scal(Array& x,
          const ScalarType alpha)
{
    using thrust::system::detail::generic::select_system;

    typedef typename Array::memory_space System;

    System system;

    cusp::blas::scal(select_system(system), x, alpha);
}

template <typename DerivedPolicy,
          typename Array,
          typename ScalarType>
void scal(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                Array& x,
          const ScalarType alpha)
{
    using cusp::blas::thrustblas::scal;

    scal(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), x, alpha);
}

template<typename Array2d,
         typename Array1,
         typename Array2>
void gemv(const Array2d& A,
          const Array1& x,
                Array2& y)
{
    using thrust::system::detail::generic::select_system;

    typedef typename Array2d::memory_space System1;
    typedef typename Array1::memory_space  System2;
    typedef typename Array2::memory_space  System3;

    System1 system1;
    System2 system2;
    System3 system3;

    cusp::blas::gemv(select_system(system1,system2,system3), A, x, y);
}

template <typename DerivedPolicy,
          typename Array2d,
          typename Array1,
          typename Array2>
void gemv(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
          const Array2d& A,
          const Array1&  x,
                Array2&  y)
{
    using cusp::blas::thrustblas::gemv;

    gemv(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), A, x, y);
}

template<typename Array2d1,
         typename Array2d2,
         typename Array2d3>
void gemm(const Array2d1& A,
          const Array2d2& B,
                Array2d3& C)
{
    using thrust::system::detail::generic::select_system;

    typedef typename Array2d1::memory_space System1;
    typedef typename Array2d2::memory_space System2;
    typedef typename Array2d3::memory_space System3;

    System1 system1;
    System2 system2;
    System3 system3;

    cusp::blas::gemm(select_system(system1,system2,system3), A, B, C);
}

template <typename DerivedPolicy,
          typename Array2d1,
          typename Array2d2,
          typename Array2d3>
void gemm(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
          const Array2d1& A,
          const Array2d2& B,
                Array2d3& C)
{
    using cusp::blas::thrustblas::gemm;

    gemm(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), A, B, C);
}

} // end namespace blas
} // end namespace cusp

