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

#include <cusp/blas/blas_policy.h>

#include <thrust/iterator/iterator_traits.h>


namespace cusp
{
namespace blas
{

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

    cusp::assert_same_dimensions(x, y);
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
    cusp::blas::axpy(policy, x, y, ValueType(alpha));
}

template <typename DerivedPolicy,
          typename Array1,
          typename Array2,
          typename ScalarType>
void axpy(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
          const Array1& x,
          Array2& y,
          ScalarType alpha)
{
    axpy(x, y, alpha);
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
    cusp::assert_same_dimensions(x, y, z);

    size_t N = x.size();
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(x.begin(), y.begin(), z.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(x.begin(), y.begin(), z.begin())) + N,
                     cusp::detail::AXPBY<ScalarType1,ScalarType2>(alpha, beta));
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
           Array3& output,
           ScalarType1 alpha,
           ScalarType2 beta)
{
    cusp::blas::axpby(x, y, output, alpha, beta);
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
    cusp::assert_same_dimensions(x, y, z, output);

    size_t N = x.size();
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(x.begin(), y.begin(), z.begin(), output.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(x.begin(), y.begin(), z.begin(), output.begin())) + N,
                     cusp::detail::AXPBYPCZ<ScalarType1,ScalarType2,ScalarType3>(alpha, beta, gamma));
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
    cusp::blas::axpbypcz(x, y, z, output, alpha, beta, gamma);
}

template <typename Array1,
          typename Array2,
          typename Array3>
void xmy(const Array1& x,
         const Array2& y,
         Array3& output)
{
    typedef typename Array3::value_type ValueType;

    cusp::assert_same_dimensions(x, y, output);
    thrust::transform(x.begin(), x.end(), y.begin(), output.begin(), cusp::detail::XMY<ValueType>());
}

template <typename DerivedPolicy,
          typename Array1,
          typename Array2,
          typename Array3>
void xmy(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
         const Array1& x,
         const Array2& y,
         Array3& output)
{
    cusp::blas::xmy(x,y,output);
}

template <typename Array1,
          typename Array2>
void copy(const Array1& x,
          Array2& y)
{
    cusp::assert_same_dimensions(x, y);
    thrust::copy(x.begin(), x.end(), y.begin());
}

template <typename DerivedPolicy,
          typename Array1,
          typename Array2>
void copy(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
          const Array1& x,
          Array2& y)
{
    cusp::blas::copy(x,y);
}

template <typename Array1,
          typename RandomAccessIterator>
void copy(const Array1& x,
          cusp::array1d_view<RandomAccessIterator> y)
{
    cusp::assert_same_dimensions(x, y);
    thrust::copy(x.begin(), x.end(), y.begin());
}

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

    return cusp::blas::dot(DerivedPolicy(), x, y);
}

template <typename Array1,
          typename Array2>
typename Array1::value_type
dot(const Array1& x,
    const Array2& y)
{
    cusp::assert_same_dimensions(x, y);

    typedef typename Array1::value_type ValueType;
    typedef typename Array1::memory_space MemorySpace;
    blas_policy<ValueType,MemorySpace> policy;
    return cusp::blas::dot(policy, x, y);
}

template <typename DerivedPolicy,
          typename Array1,
          typename Array2>
typename Array1::value_type
dot(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
    const Array1& x,
    const Array2& y)
{
    return cusp::blas::dot(x, y);
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

    return cusp::blas::dotc(DerivedPolicy(), x, y);
}

template <typename Array1,
          typename Array2>
typename Array1::value_type
dotc(const Array1& x,
     const Array2& y)
{
    cusp::assert_same_dimensions(x, y);

    typedef typename Array1::value_type ValueType;
    typedef typename Array1::memory_space MemorySpace;
    blas_policy<ValueType,MemorySpace> policy;
    return cusp::blas::dotc(policy, x, y);
}

template <typename DerivedPolicy,
          typename Array1,
          typename Array2>
typename Array1::value_type
dotc(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
     const Array1& x,
     const Array2& y)
{
    return dotc(x,y);
}

template <typename Array,
          typename ScalarType>
typename thrust::detail::enable_if_convertible<typename Array::format,cusp::array1d_format>::type
fill(Array& x,
     const ScalarType alpha)
{
    thrust::fill(x.begin(), x.end(), alpha);
}

template <typename RandomAccessIterator,
          typename ScalarType>
void fill(cusp::array1d_view<RandomAccessIterator> x,
          const ScalarType alpha)
{
    thrust::fill(x.begin(), x.end(), alpha);
}

template <typename DerivedPolicy,
          typename Array,
          typename ScalarType>
void fill(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
          Array& x,
          ScalarType alpha)
{
    cusp::blas::fill(x, alpha);
}

template <typename Array>
typename cusp::detail::norm_type<typename Array::value_type>::type
nrm1(const blas_policy<typename Array::value_type,typename Array::memory_space>& policy,
     const Array& x)
{
    typedef typename Array::value_type ValueType;
    typedef typename Array::memory_space MemorySpace;
    typedef typename blas_policy<ValueType,MemorySpace>::type DerivedPolicy;

    return cusp::blas::nrm1(DerivedPolicy(), x);
}

template <typename Array>
typename cusp::detail::norm_type<typename Array::value_type>::type
nrm1(const Array& x)
{
    typedef typename Array::value_type ValueType;
    typedef typename Array::memory_space MemorySpace;
    blas_policy<ValueType,MemorySpace> policy;
    return cusp::blas::nrm1(policy, x);
}

template <typename DerivedPolicy,
          typename Array>
typename cusp::detail::norm_type<typename Array::value_type>::type
nrm1(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
     const Array& x)
{
    return cusp::blas::nrm1(x);
}

template <typename Array>
typename cusp::detail::norm_type<typename Array::value_type>::type
nrm2(const blas_policy<typename Array::value_type,typename Array::memory_space>& policy,
     const Array& x)
{
    typedef typename Array::value_type ValueType;
    typedef typename Array::memory_space MemorySpace;
    typedef typename blas_policy<ValueType,MemorySpace>::type DerivedPolicy;

    return cusp::blas::nrm2(DerivedPolicy(), x);
}

template <typename Array>
typename cusp::detail::norm_type<typename Array::value_type>::type
nrm2(const Array& x)
{
    typedef typename Array::value_type ValueType;
    typedef typename Array::memory_space MemorySpace;
    blas_policy<ValueType,MemorySpace> policy;

    return cusp::blas::nrm2(policy, x);
}

template <typename DerivedPolicy,
          typename Array>
typename cusp::detail::norm_type<typename Array::value_type>::type
nrm2(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
     const Array& x)
{
    return cusp::blas::nrm2(x);
}

template <typename Array>
typename Array::value_type
nrmmax(const blas_policy<typename Array::value_type,typename Array::memory_space>& policy,
       const Array& x)
{
    typedef typename Array::value_type ValueType;
    typedef typename Array::memory_space MemorySpace;
    typedef typename blas_policy<ValueType,MemorySpace>::type DerivedPolicy;

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

template <typename DerivedPolicy,
          typename Array>
typename Array::value_type
nrmmax(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
       const Array& x)
{
    return cusp::blas::nrmmax(x);
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
    cusp::blas::scal(policy, x, ValueType(alpha));
}

template <typename RandomAccessIterator,
          typename ScalarType>
void scal(cusp::array1d_view<RandomAccessIterator> x,
          ScalarType alpha)
{
    thrust::for_each(x.begin(), x.end(), cusp::detail::SCAL<ScalarType>(alpha));
}

template <typename DerivedPolicy,
          typename Array,
          typename ScalarType>
void scal(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
          Array& x,
          ScalarType alpha)
{
    cusp::blas::scal(x, alpha);
}

template<typename Array2d,
         typename Array1,
         typename Array2>
void gemv(const blas_policy<typename Array2::value_type,typename Array2::memory_space>& policy,
          const Array2d& A,
          const Array1& x,
          Array2& y)
{
    typedef typename Array2::value_type ValueType;
    typedef typename Array2::memory_space MemorySpace;
    typedef typename blas_policy<ValueType,MemorySpace>::type DerivedPolicy;

    cusp::blas::gemv(DerivedPolicy(), A, x, y);
}

template<typename Array2d,
         typename Array1,
         typename Array2>
void gemv(const Array2d& A,
          const Array1& x,
          Array2& y)
{
    typedef typename Array2::value_type ValueType;
    typedef typename Array2::memory_space MemorySpace;
    blas_policy<ValueType,MemorySpace> policy;
    cusp::blas::gemv(policy, A, x, y);
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
    cusp::blas::gemv(A, x, y);
}

template<typename Array2d1,
         typename Array2d2,
         typename Array2d3>
void gemm(const blas_policy<typename Array2d3::value_type,typename Array2d3::memory_space>& policy,
          const Array2d1& A,
          const Array2d2& B,
          Array2d3& C)
{
    typedef typename Array2d3::value_type ValueType;
    typedef typename Array2d3::memory_space MemorySpace;
    typedef typename blas_policy<ValueType,MemorySpace>::type DerivedPolicy;

    cusp::blas::gemm(DerivedPolicy(), A, B, C);
}

template<typename Array2d1,
         typename Array2d2,
         typename Array2d3>
void gemm(const Array2d1& A,
          const Array2d2& B,
          Array2d3& C)
{
    typedef typename Array2d3::value_type ValueType;
    typedef typename Array2d3::memory_space MemorySpace;
    blas_policy<ValueType,MemorySpace> policy;
    gemm(policy, A, B, C);
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
    cusp::blas::gemm(A,B,C);
}

} // end namespace blas
} // end namespace cusp


