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

/*! \file blas.h
 *  \brief BLAS-like functions
 */


#pragma once

#include <cusp/detail/config.h>

#include <cusp/complex.h>

#include <thrust/iterator/iterator_traits.h>

namespace cusp
{
namespace blas
{

/*! \addtogroup algorithms Algorithms
 */

/*! \addtogroup blas BLAS
 *  \ingroup algorithms
 *  \{
 */

/*! \p axpy : scaled vector addition (y = alpha * x + y)
 */
template <typename Array1,
          typename Array2,
          typename ScalarType>
void axpy(const Array1& x,
                Array2& y,
          ScalarType alpha);

template <typename Array1,
          typename Array2,
          typename ScalarType>
void axpy(const Array1& x,
          const Array2& y,
          ScalarType alpha);

/*! \p axpby : linear combination of two vectors (output = alpha * x + beta * y)
 */
template <typename Array1,
          typename Array2,
          typename Array3,
          typename ScalarType1,
          typename ScalarType2>
void axpby(const Array1& x,
           const Array2& y,
                 Array3& output,
           ScalarType1 alpha,
           ScalarType2 beta);

template <typename Array1,
          typename Array2,
          typename Array3,
          typename ScalarType1,
          typename ScalarType2>
void axpby(const Array1& x,
           const Array2& y,
           const Array3& output,
           ScalarType1 alpha,
           ScalarType2 beta);

/*! \p axpbycz : linear combination of three vectors (output = alpha * x + beta * y + gamma * z)
 */
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
              ScalarType3 gamma);

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
              ScalarType3 gamma);

/*! \p xmy : elementwise multiplication of two vectors (output[i] = x[i] * y[i])
 */
template <typename Array1,
          typename Array2,
          typename Array3>
void xmy(const Array1& x,
         const Array2& y,
               Array3& output);

template <typename Array1,
          typename Array2,
          typename Array3>
void xmy(const Array1& x,
         const Array2& y,
         const Array3& output);

/*! \p copy : vector copy (y = x)
 */
template <typename Array1,
          typename Array2>
void copy(const Array1& array1,
                Array2& array2);

template <typename Array1,
          typename Array2>
void copy(const Array1& array1,
          const Array2& array2);

/*! \p dot : dot product (x^T * y)
 */
template <typename Array1,
          typename Array2>
typename Array1::value_type
    dot(const Array1& x,
        const Array2& y);

/*! \p dotc : conjugate dot product (conjugate(x)^T * y)
 */
template <typename Array1,
          typename Array2>
typename Array1::value_type
    dotc(const Array1& x,
         const Array2& y);

/*! \p fill : vector fill (x[i] = alpha)
 */
template <typename Array,
          typename ScalarType>
void fill(Array& array,
          ScalarType alpha);

template <typename Array,
          typename ScalarType>
void fill(const Array& array,
          ScalarType alpha);

/*! \p nrm1 : vector 1-norm (sum abs(x[i]))
 */
template <typename Array>
typename norm_type<typename Array::value_type>::type
    nrm1(const Array& array);

/*! \p nrm2 : vector 2-norm (sqrt(sum x[i] * x[i] )
 */
template <typename Array>
typename norm_type<typename Array::value_type>::type
    nrm2(const Array& array);

/*! \p nrmmax : vector infinity norm
 */
template <typename Array>
typename Array::value_type
    nrmmax(const Array& array);

/*! \p scal : scale vector (x[i] = alpha * x[i])
 */
template <typename Array,
          typename ScalarType>
void scal(Array& x,
          ScalarType alpha);

template <typename Array,
          typename ScalarType>
void scal(const Array& x,
          ScalarType alpha);

template<typename Array2d,
         typename Array1,
         typename Array2>
void gemv(const Array2d& A,
          const Array1& x,
          Array2& y);

template<typename Array2d1,
         typename Array2d2,
         typename Array2d3>
void gemm(const Array2d1& A,
          const Array2d2& B,
          Array2d3& C);
/*! \}
 */

} // end namespace blas
} // end namespace cusp

#include <cusp/blas/blas.inl>
