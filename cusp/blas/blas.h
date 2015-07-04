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

/*! \file blas.h
 *  \brief BLAS-like functions
 */


#pragma once

#include <cusp/detail/config.h>

#include <cusp/complex.h>
#include <cusp/detail/type_traits.h>

namespace cusp
{
namespace blas
{

/*! \addtogroup dense Dense Algorithms
 *  \addtogroup blas BLAS
 *  \ingroup dense
 *  \brief Interface to BLAS routines
 *  \{
 */

/*! \cond */
template <typename DerivedPolicy,
          typename ArrayType>
int amax(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
         const ArrayType& x);
/*! \endcond */

/**
 * \brief index of the largest element in a array
 *
 * \tparam ArrayType Type of the input array
 *
 * \param x The input array to find max value
 *
 * \par Example
 * \code
 * #include <cusp/array1d.h>
 * #include <cusp/print.h>
 *
 * // include cusp blas header file
 * #include <cusp/blas/blas.h>
 *
 * int main()
 * {
 *   // create an array
 *   cusp::array1d<float,cusp::host_memory> x(10);
 *
 *   // fill x array with random values
 *   cusp::random_array<float> rand(10);
 *
 *   // find index of max absolute value in x
 *   int index = cusp::blas::amax(x);
 *
 *   std::cout << "Max value at pos: " << index << std::endl;
 *
 *   return 0;
 * }
 * \endcode
 */
template <typename ArrayType>
int amax(const ArrayType& x);

/*! \cond */
template <typename DerivedPolicy,
          typename Array>
typename cusp::detail::norm_type<typename Array::value_type>::type
asum(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
     const Array& x);
/*! \endcond */

/**
 * \brief sum of absolute value of all entries in array
 *
 * \tparam ArrayType Type of the input array
 *
 * \param x The input array to compute sum of absolute values
 *
 * \par Example
 * \code
 * #include <cusp/array1d.h>
 * #include <cusp/print.h>
 *
 * // include cusp blas header file
 * #include <cusp/blas/blas.h>
 *
 * int main()
 * {
 *   // create an array
 *   cusp::array1d<float,cusp::host_memory> x(10);
 *
 *   // fill x array with random values
 *   cusp::random_array<float> rand(10);
 *
 *   // find index of max absolute value in x
 *   float sum = cusp::blas::asum(x);
 *
 *   std::cout << "asum(x) =" << sum << std::endl;
 *
 *   return 0;
 * }
 * \endcode
 */
template <typename ArrayType>
typename cusp::detail::norm_type<typename ArrayType::value_type>::type
asum(const ArrayType& x);

/*! \cond */
template <typename DerivedPolicy,
          typename ArrayType1,
          typename ArrayType2,
          typename ScalarType>
void axpy(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
          const ArrayType1& x,
                ArrayType2& y,
          const ScalarType alpha);
/*! \endcond */

/**
 * \brief scaled vector addition (y = alpha * x + y)
 *
 * \tparam ArrayType Type of the input array
 *
 * \param x The input array to compute sum of absolute values
 *
 * \par Example
 * \code
 * #include <cusp/array1d.h>
 * #include <cusp/print.h>
 *
 * // include cusp blas header file
 * #include <cusp/blas/blas.h>
 *
 * int main()
 * {
 *   // create an array
 *   cusp::array1d<float,cusp::host_memory> x(10);
 *
 *   // fill x array with random values
 *   cusp::random_array<float> rand(10);
 *
 *   // find index of max absolute value in x
 *   float sum = cusp::blas::asum(x);
 *
 *   std::cout << "asum(x) =" << sum << std::endl;
 *
 *   return 0;
 * }
 * \endcode
 */
template <typename ArrayType1,
          typename ArrayType2,
          typename ScalarType>
void axpy(const ArrayType1& x,
                ArrayType2& y,
          const ScalarType alpha);

/*! \cond */
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
                 ScalarType2 beta);
/*! \endcond */

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

/*! \cond */
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
              ScalarType3 gamma);
/*! \endcond */

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

/*! \cond */
template <typename DerivedPolicy,
          typename Array1,
          typename Array2,
          typename Array3>
void xmy(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
         const Array1& x,
         const Array2& y,
               Array3& output);
/*! \endcond */

/*! \p xmy : elementwise multiplication of two vectors (output[i] = x[i] * y[i])
 */
template <typename Array1,
          typename Array2,
          typename Array3>
void xmy(const Array1& x,
         const Array2& y,
               Array3& output);

/*! \cond */
template <typename DerivedPolicy,
          typename Array1,
          typename Array2>
void copy(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
          const Array1& x,
                Array2& y);
/*! \endcond */

/*! \p copy : vector copy (y = x)
 */
template <typename Array1,
          typename Array2>
void copy(const Array1& x,
                Array2& y);

/*! \cond */
template <typename DerivedPolicy,
          typename Array1,
          typename Array2>
typename Array1::value_type
dot(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
    const Array1& x,
    const Array2& y);
/*! \endcond */

/*! \p dot : dot product (x^T * y)
 */
template <typename Array1,
          typename Array2>
typename Array1::value_type
dot(const Array1& x,
    const Array2& y);

/*! \cond */
template <typename DerivedPolicy,
          typename Array1,
          typename Array2>
typename Array1::value_type
dotc(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
     const Array1& x,
     const Array2& y);
/*! \endcond */

/*! \p dotc : conjugate dot product (conjugate(x)^T * y)
 */
template <typename Array1,
          typename Array2>
typename Array1::value_type
dotc(const Array1& x,
     const Array2& y);

/*! \cond */
template <typename DerivedPolicy,
          typename ArrayType,
          typename ScalarType>
void fill(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
          ArrayType& array,
          const ScalarType alpha);
/*! \endcond */

/**
 * \brief vector fill (x[i] = alpha)
 *
 * \tparam ArrayType Type of the input array
 * \tparam ScalarType Type of the fill value
 *
 * \param x The input array to fill
 * \param alpha Value to fill array x
 *
 * \par Example
 * \code
 * #include <cusp/array1d.h>
 * #include <cusp/print.h>
 *
 * // include cusp blas header file
 * #include <cusp/blas/blas.h>
 *
 * int main()
 * {
 *   // create an array
 *   cusp::array1d<float,cusp::host_memory> x(10);
 *
 *   // fill x array with 1s
 *   cusp::blas::fill(x, 1);
 *
 *   return 0;
 * }
 * \endcode
 */
template <typename ArrayType,
          typename ScalarType>
void fill(ArrayType& x,
          const ScalarType alpha);

/*! \cond */
template <typename DerivedPolicy,
          typename Array>
typename cusp::detail::norm_type<typename Array::value_type>::type
nrm1(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
     const Array& array);
/*! \endcond */

/**
 * \brief vector 1-norm (sum abs(x[i]))
 *
 * \tparam ArrayType Type of the input array
 *
 * \param x The input array to find 2-norm
 *
 * \par Example
 * \code
 * #include <cusp/array1d.h>
 * #include <cusp/print.h>
 *
 * // include cusp blas header file
 * #include <cusp/blas/blas.h>
 *
 * int main()
 * {
 *   // create an array initially filled with random values
 *   cusp::array1d<float,cusp::host_memory> x(10);
 *   cusp::random_array<float> rand(10);
 *   cusp::blas::copy(rand, x);
 *
 *   // compute and print 1-norm
 *   float nrm_x = cusp::blas::nrm1(x);
 *   std::cout << "nrm1(x) = " << nrm_x << std::endl;
 *
 *   return 0;
 * }
 * \endcode
 */
template <typename ArrayType>
typename cusp::detail::norm_type<typename ArrayType::value_type>::type
nrm1(const ArrayType& x);

/*! \cond */
template <typename DerivedPolicy,
          typename ArrayType>
typename cusp::detail::norm_type<typename ArrayType::value_type>::type
nrm2(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
     const ArrayType& x);
/*! \endcond */

/**
 * \brief vector 2-norm (sqrt(sum x[i] * x[i] )
 *
 * \tparam ArrayType Type of the input array
 *
 * \param x The input array to find 2-norm
 *
 * \par Example
 * \code
 * #include <cusp/array1d.h>
 * #include <cusp/print.h>
 *
 * // include cusp blas header file
 * #include <cusp/blas/blas.h>
 *
 * int main()
 * {
 *   // create an array initially filled with random values
 *   cusp::array1d<float,cusp::host_memory> x(10);
 *   cusp::random_array<float> rand(10);
 *   cusp::blas::copy(rand, x);
 *
 *   // compute and print 2-norm
 *   float nrm_x = cusp::blas::nrm2(x);
 *   std::cout << "nrm2(x) = " << nrm_x << std::endl;
 *
 *   return 0;
 * }
 * \endcode
 */
template <typename ArrayType>
typename cusp::detail::norm_type<typename ArrayType::value_type>::type
nrm2(const ArrayType& x);

/*! \cond */
template <typename DerivedPolicy,
          typename ArrayType>
typename ArrayType::value_type
nrmmax(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
       const ArrayType& x);
/*! \endcond */

/**
 * \brief vector infinity norm
 *
 * \tparam ArrayType Type of the input array
 *
 * \param x The input array to find infinity norm
 *
 * \par Example
 * \code
 * #include <cusp/array1d.h>
 * #include <cusp/print.h>
 *
 * // include cusp blas header file
 * #include <cusp/blas/blas.h>
 *
 * int main()
 * {
 *   // create an array initially filled with random values
 *   cusp::array1d<float,cusp::host_memory> x(10);
 *   cusp::random_array<float> rand(10);
 *   cusp::blas::copy(rand, x);
 *
 *   // compute and print infinity norm
 *   float nrm_x = cusp::blas::nrmmax(x);
 *   std::cout << "nrmmax(x) = " << nrm_x << std::endl;
 *
 *   return 0;
 * }
 * \endcode
 */
template <typename ArrayType>
typename ArrayType::value_type
nrmmax(const ArrayType& x);

/*! \cond */
template <typename DerivedPolicy,
          typename ArrayType,
          typename ScalarType>
void scal(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
          ArrayType& x,
          const ScalarType alpha);
/*! \endcond */

/**
 * \brief scale vector (x[i] = alpha * x[i])
 *
 * \tparam ArrayType  Type of the input array
 * \tparam ScalarType Type of the scalar value
 *
 * \param x The input array to scale
 * \param alpha The scale factor
 *
 * \par Example
 * \code
 * #include <cusp/array1d.h>
 * #include <cusp/print.h>
 *
 * // include cusp blas header file
 * #include <cusp/blas/blas.h>
 *
 * int main()
 * {
 *   // create an array initially filled with 2s
 *   cusp::array1d<float,cusp::host_memory> x(10, 2);
 *
 *   // scal x by 2
 *   cusp::blas::scal(x, 2);
 *
 *   // print the scaled vector
 *   cusp::print(x);
 *
 *   return 0;
 * }
 * \endcode
 */
template <typename ArrayType,
          typename ScalarType>
void scal(ArrayType& x,
          const ScalarType alpha);

/*! \cond */
template <typename DerivedPolicy,
          typename Array2d1,
          typename Array1d1,
          typename Array1d2>
void gemv(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
          const Array2d1& A,
          const Array1d1& x,
                Array1d2& y);
/*! \endcond */

/**
 * \brief Solve a triangular matrix equation
 *
 * \tparam Array2d Type of the first input matrix
 * \tparam Array1d Type of the output matrix
 *
 * \param A Contains the upper or lower triangle of a symmetric matrix
 * \param x Contains block of right-hand side vectors
 *
 * \par Example
 * \code
 * #include <cusp/array1d.h>
 * #include <cusp/array2d.h>
 * #include <cusp/print.h>
 *
 * #include <cusp/gallery/poisson.h>
 *
 * // include cusp blas header file
 * #include <cusp/blas/blas.h>
 *
 * int main()
 * {
 *   // create an empty dense matrix structure
 *   cusp::array2d<float,cusp::host_memory> A;
 *
 *   // create 2D Poisson problem
 *   cusp::gallery::poisson5pt(A, 4, 4);
 *
 *   // create an random dense array
 *   cusp::random_array<float> rand(A.num_rows);
 *   cusp::array1d<float,cusp::host_memory> x(rand);
 *
 *   // create an empty output array
 *   cusp::array1d<float,cusp::host_memory> y(A.num_rows);
 *
 *   // solve multiple RHS vectors
 *   cusp::blas::gemv(A, x, y);
 *
 *   // print the contents of y
 *   cusp::print(y);
 *
 *   return 0;
 * }
 * \endcode
 */
template<typename Array2d1,
         typename Array1d1,
         typename Array1d2>
void gemv(const Array2d1& A,
          const Array1d1& x,
                Array1d2& y);

/*! \cond */
template <typename DerivedPolicy,
          typename Array1d1,
          typename Array1d2,
          typename Array2d1>
void ger(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
         const Array1d1& x,
         const Array1d2& y,
               Array2d1& A);
/*! \endcond */

/**
 * \brief Solve a triangular matrix equation
 *
 * \tparam Array2d Type of the first input matrix
 * \tparam Array1d Type of the output matrix
 *
 * \param A Contains the upper or lower triangle of a symmetric matrix
 * \param x Contains block of right-hand side vectors
 *
 * \par Example
 * \code
 * #include <cusp/array1d.h>
 * #include <cusp/array2d.h>
 * #include <cusp/print.h>
 *
 * #include <cusp/gallery/poisson.h>
 *
 * // include cusp blas header file
 * #include <cusp/blas/blas.h>
 *
 * int main()
 * {
 *   // create an empty dense matrix structure
 *   cusp::array2d<float,cusp::host_memory> A(10,10);
 *
 *   // create 2 random dense arrays
 *   cusp::random_array<float> rand1(A.num_rows, 0);
 *   cusp::array1d<float,cusp::host_memory> x(rand1);
 *
 *   cusp::random_array<float> rand2(A.num_rows, 7);
 *   cusp::array1d<float,cusp::host_memory> y(rand2);
 *
 *   // solve multiple RHS vectors
 *   cusp::blas::ger(x, y, A);
 *
 *   // print the contents of A
 *   cusp::print(A);
 *
 *   return 0;
 * }
 * \endcode
 */
template<typename Array1d1,
         typename Array1d2,
         typename Array2d1>
void ger(const Array1d1& x,
         const Array1d2& y,
               Array2d1& A);

/*! \cond */
template <typename DerivedPolicy,
          typename Array2d1,
          typename Array1d1,
          typename Array1d2>
void symv(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
          const Array2d1& A,
          const Array1d1& x,
                Array1d2& y);
/*! \endcond */

/**
 * \brief Solve a triangular matrix equation
 *
 * \tparam Array2d Type of the first input matrix
 * \tparam Array1d Type of the output matrix
 *
 * \param A Contains the upper or lower triangle of a symmetric matrix
 * \param x Contains block of right-hand side vectors
 *
 * \par Example
 * \code
 * #include <cusp/array1d.h>
 * #include <cusp/array2d.h>
 * #include <cusp/print.h>
 *
 * #include <cusp/gallery/poisson.h>
 *
 * // include cusp blas header file
 * #include <cusp/blas/blas.h>
 *
 * int main()
 * {
 *   // create an empty dense matrix structure
 *   cusp::array2d<float,cusp::host_memory> A;
 *
 *   // create 2D Poisson problem
 *   cusp::gallery::poisson5pt(A, 4, 4);
 *
 *   // create an random dense array
 *   cusp::random_array<float> rand(A.num_rows);
 *   cusp::array1d<float,cusp::host_memory> x(rand);
 *
 *   // create an empty output array
 *   cusp::array1d<float,cusp::host_memory> y(A.num_rows);
 *
 *   // solve multiple RHS vectors
 *   cusp::blas::symv(A, x, y);
 *
 *   // print the contents of y
 *   cusp::print(y);
 *
 *   return 0;
 * }
 * \endcode
 */
template <typename Array2d1,
          typename Array1d1,
          typename Array1d2>
void symv(const Array2d1& A,
          const Array1d1& x,
                Array1d2& y);

/*! \cond */
template <typename DerivedPolicy,
          typename Array1d,
          typename Array2d>
void syr(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
         const Array1d& x,
               Array2d& A);
/*! \endcond */

/**
 * \brief Solve a triangular matrix equation
 *
 * \tparam Array2d Type of the first input matrix
 * \tparam Array1d Type of the output matrix
 *
 * \param A Contains the upper or lower triangle of a symmetric matrix
 * \param x Contains block of right-hand side vectors
 *
 * \par Example
 * \code
 * #include <cusp/array1d.h>
 * #include <cusp/array2d.h>
 * #include <cusp/print.h>
 *
 * #include <cusp/gallery/poisson.h>
 *
 * // include cusp blas header file
 * #include <cusp/blas/blas.h>
 *
 * int main()
 * {
 *   // create an empty dense matrix structure
 *   cusp::array2d<float,cusp::host_memory> A(10,10);
 *
 *   // create an random dense array
 *   cusp::random_array<float> rand(A.num_rows);
 *   cusp::array1d<float,cusp::host_memory> x(rand);
 *
 *   // solve multiple RHS vectors
 *   cusp::blas::syr(x, A);
 *
 *   // print the contents of A
 *   cusp::print(A);
 *
 *   return 0;
 * }
 * \endcode
 */
template <typename Array1d,
          typename Array2d>
void syr(const Array1d& x,
               Array2d& A);

/*! \cond */
template <typename DerivedPolicy,
          typename Array2d,
          typename Array1d>
void trmv(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
          const Array2d& A,
                Array1d& x);
/*! \endcond */

/**
 * \brief Solve a triangular matrix equation
 *
 * \tparam Array2d Type of the first input matrix
 * \tparam Array1d Type of the output matrix
 *
 * \param A Contains the upper or lower triangle of a symmetric matrix
 * \param x Contains block of right-hand side vectors
 *
 * \par Example
 * \code
 * #include <cusp/array1d.h>
 * #include <cusp/array2d.h>
 * #include <cusp/print.h>
 *
 * #include <cusp/gallery/poisson.h>
 *
 * // include cusp blas header file
 * #include <cusp/blas/blas.h>
 *
 * int main()
 * {
 *   // create an empty dense matrix structure
 *   cusp::array2d<float,cusp::host_memory> A;
 *
 *   // create 2D Poisson problem
 *   cusp::gallery::poisson5pt(A, 4, 4);
 *
 *   // create an random dense array
 *   cusp::random_array<float> rand(A.num_rows);
 *   cusp::array1d<float,cusp::host_memory> x(rand);
 *
 *   // solve multiple RHS vectors
 *   cusp::blas::trmv(A, x);
 *
 *   // print the contents of x
 *   cusp::print(x);
 *
 *   return 0;
 * }
 * \endcode
 */
template<typename Array2d,
         typename Array1d>
void trmv(const Array2d& A,
                Array1d& x);

/*! \cond */
template <typename DerivedPolicy,
          typename Array2d,
          typename Array1d>
void trsv(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
          const Array2d& A,
                Array1d& x);
/*! \endcond */

/**
 * \brief Solve a triangular matrix equation
 *
 * \tparam Array2d Type of the first input matrix
 * \tparam Array1d Type of the output matrix
 *
 * \param A Contains the upper or lower triangle of a symmetric matrix
 * \param x Contains block of right-hand side vectors
 *
 * \par Example
 * \code
 * #include <cusp/array1d.h>
 * #include <cusp/array2d.h>
 * #include <cusp/print.h>
 *
 * #include <cusp/gallery/poisson.h>
 *
 * // include cusp blas header file
 * #include <cusp/blas/blas.h>
 *
 * int main()
 * {
 *   // create an empty dense matrix structure
 *   cusp::array2d<float,cusp::host_memory> A;
 *
 *   // create 2D Poisson problem
 *   cusp::gallery::poisson5pt(A, 4, 4);
 *
 *   // create an random dense array
 *   cusp::random_array<float> rand(A.num_rows);
 *   cusp::array1d<float,cusp::host_memory> x(rand);
 *
 *   // solve multiple RHS vectors
 *   cusp::blas::trsv(A, x);
 *
 *   // print the contents of x
 *   cusp::print(x);
 *
 *   return 0;
 * }
 * \endcode
 */
template<typename Array2d,
         typename Array1d>
void trsv(const Array2d& A,
                Array1d& x);

/*! \cond */
template <typename DerivedPolicy,
          typename Array2d1,
          typename Array2d2,
          typename Array2d3>
void gemm(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
          const Array2d1& A,
          const Array2d2& B,
                Array2d3& C);
/*! \endcond */

/**
 * \brief Solve a triangular matrix equation
 *
 * \tparam Array2d1 Type of the first input matrix
 * \tparam Array2d2 Type of the second input matrix
 * \tparam Array2d3 Type of the output matrix
 *
 * \param A Contains the upper or lower triangle of a symmetric matrix
 * \param B Contains the upper or lower triangle of a symmetric matrix
 * \param C Contains block of right-hand side vectors
 *
 * \par Example
 * \code
 * #include <cusp/array2d.h>
 * #include <cusp/print.h>
 *
 * #include <cusp/gallery/poisson.h>
 *
 * // include cusp blas header file
 * #include <cusp/blas/blas.h>
 *
 * int main()
 * {
 *   // create an empty dense matrix structure
 *   cusp::array2d<float,cusp::host_memory> A, B;
 *
 *   // create 2D Poisson problem
 *   cusp::gallery::poisson5pt(A, 4, 4);
 *
 *   // solve multiple RHS vectors
 *   cusp::blas::gemm(A, A, B);
 *
 *   // print the contents of B
 *   cusp::print(B);
 *
 *   return 0;
 * }
 * \endcode
 */
template<typename Array2d1,
         typename Array2d2,
         typename Array2d3>
void gemm(const Array2d1& A,
          const Array2d2& B,
                Array2d3& C);

/*! \cond */
template <typename DerivedPolicy,
          typename Array2d1,
          typename Array2d2,
          typename Array2d3>
void symm(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
          const Array2d1& A,
          const Array2d2& B,
                Array2d3& C);
/*! \endcond */

/**
 * \brief Solve a triangular matrix equation
 *
 * \tparam Array2d1 Type of the first input matrix
 * \tparam Array2d2 Type of the second input matrix
 * \tparam Array2d3 Type of the output matrix
 *
 * \param A Contains the upper or lower triangle of a symmetric matrix
 * \param B Contains the upper or lower triangle of a symmetric matrix
 * \param C Contains block of right-hand side vectors
 *
 * \par Example
 * \code
 * #include <cusp/array2d.h>
 * #include <cusp/print.h>
 *
 * #include <cusp/gallery/poisson.h>
 *
 * // include cusp blas header file
 * #include <cusp/blas/blas.h>
 *
 * int main()
 * {
 *   // create an empty dense matrix structure
 *   cusp::array2d<float,cusp::host_memory> A, B;
 *
 *   // create 2D Poisson problem
 *   cusp::gallery::poisson5pt(A, 4, 4);
 *
 *   // solve multiple RHS vectors
 *   cusp::blas::symm(A, A, B);
 *
 *   // print the contents of B
 *   cusp::print(B);
 *
 *   return 0;
 * }
 * \endcode
 */
template<typename Array2d1,
         typename Array2d2,
         typename Array2d3>
void symm(const Array2d1& A,
          const Array2d2& B,
                Array2d3& C);

/*! \cond */
template <typename DerivedPolicy,
          typename Array2d1,
          typename Array2d2>
void syrk(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
          const Array2d1& A,
                Array2d2& B);
/*! \endcond */

/**
 * \brief Solve a triangular matrix equation
 *
 * \tparam Array2d1 Type of the first input matrix
 * \tparam Array2d2 Type of the output matrix
 *
 * \param A Contains the upper or lower triangle of a symmetric matrix
 * \param B Contains block of right-hand side vectors
 *
 * \par Example
 * \code
 * #include <cusp/array2d.h>
 * #include <cusp/print.h>
 *
 * #include <cusp/gallery/poisson.h>
 *
 * // include cusp blas header file
 * #include <cusp/blas/blas.h>
 *
 * int main()
 * {
 *   // create an empty dense matrix structure
 *   cusp::array2d<float,cusp::host_memory> A, B;
 *
 *   // create 2D Poisson problem
 *   cusp::gallery::poisson5pt(A, 4, 4);
 *
 *   // solve multiple RHS vectors
 *   cusp::blas::syrk(A, B);
 *
 *   // print the contents of B
 *   cusp::print(B);
 *
 *   return 0;
 * }
 * \endcode
 */
template<typename Array2d1,
         typename Array2d2>
void syrk(const Array2d1& A,
                Array2d2& B);

/*! \cond */
template <typename DerivedPolicy,
          typename Array2d1,
          typename Array2d2,
          typename Array2d3>
void syr2k(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
           const Array2d1& A,
           const Array2d2& B,
                 Array2d3& C);
/*! \endcond */

/**
 * \brief Solve a triangular matrix equation
 *
 * \tparam Array2d1 Type of the first input matrix
 * \tparam Array2d2 Type of the second input matrix
 * \tparam Array2d3 Type of the output matrix
 *
 * \param A Contains the upper or lower triangle of a symmetric matrix
 * \param B Contains the upper or lower triangle of a symmetric matrix
 * \param C Contains block of right-hand side vectors
 *
 * \par Example
 * \code
 * #include <cusp/array2d.h>
 * #include <cusp/print.h>
 *
 * #include <cusp/gallery/poisson.h>
 *
 * // include cusp blas header file
 * #include <cusp/blas/blas.h>
 *
 * int main()
 * {
 *   // create an empty dense matrix structure
 *   cusp::array2d<float,cusp::host_memory> A, B;
 *
 *   // create 2D Poisson problem
 *   cusp::gallery::poisson5pt(A, 4, 4);
 *
 *   // solve multiple RHS vectors
 *   cusp::blas::syr2k(A, A, B);
 *
 *   // print the contents of B
 *   cusp::print(B);
 *
 *   return 0;
 * }
 * \endcode
 */
template<typename Array2d1,
         typename Array2d2,
         typename Array2d3>
void syr2k(const Array2d1& A,
           const Array2d2& B,
                 Array2d3& C);

/*! \cond */
template <typename DerivedPolicy,
          typename Array2d1,
          typename Array2d2>
void trmm(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
          const Array2d1& A,
                Array2d2& B);
/*! \endcond */

/**
 * \brief Solve a triangular matrix equation
 *
 * \tparam Array2d1 Type of the first input matrix
 * \tparam Array2d2 Type of the output matrix
 *
 * \param A Contains the upper or lower triangle of a symmetric matrix
 * \param B Contains block of right-hand side vectors
 *
 * \par Example
 * \code
 * #include <cusp/array2d.h>
 * #include <cusp/print.h>
 *
 * #include <cusp/gallery/poisson.h>
 *
 * // include cusp blas header file
 * #include <cusp/blas/blas.h>
 *
 * int main()
 * {
 *   // create an empty dense matrix structure
 *   cusp::array2d<float,cusp::host_memory> A, B;
 *
 *   // create 2D Poisson problem
 *   cusp::gallery::poisson5pt(A, 4, 4);
 *
 *   // solve multiple RHS vectors
 *   cusp::blas::trmm(A, B);
 *
 *   // print the contents of B
 *   cusp::print(B);
 *
 *   return 0;
 * }
 * \endcode
 */
template<typename Array2d1,
         typename Array2d2>
void trmm(const Array2d1& A,
                Array2d2& B);

/*! \cond */
template <typename DerivedPolicy,
          typename Array2d1,
          typename Array2d2>
void trsm(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
          const Array2d1& A,
                Array2d2& B);
/*! \endcond */

/**
 * \brief Solve a triangular matrix equation
 *
 * \tparam Array2d1 Type of the first input matrix
 * \tparam Array2d2 Type of the output matrix
 *
 * \param A Contains the upper or lower triangle of a symmetric matrix
 * \param B Contains block of right-hand side vectors
 *
 * \par Example
 * \code
 * #include <cusp/array2d.h>
 * #include <cusp/print.h>
 *
 * #include <cusp/gallery/poisson.h>
 *
 * // include cusp blas header file
 * #include <cusp/blas/blas.h>
 *
 * int main()
 * {
 *   // create an empty dense matrix structure
 *   cusp::array2d<float,cusp::host_memory> A;
 *
 *   // create 2D Poisson problem
 *   cusp::gallery::poisson5pt(A, 4, 4);
 *
 *   // create a set of random RHS vectors
 *   cusp::array2d<float,cusp::host_memory> B(A.num_rows, 5);
 *
 *   // fill B with random values
 *   cusp::random_array<float> rand(B.num_entries);
 *   cusp::blas::copy(rand, B.values);
 *
 *   // solve multiple RHS vectors
 *   cusp::blas::trsm(A, B);
 *
 *   // print the contents of B
 *   cusp::print(B);
 *
 *   return 0;
 * }
 * \endcode
 */
template<typename Array2d1,
         typename Array2d2>
void trsm(const Array2d1& A,
                Array2d2& B);

/*! \}
 */

} // end namespace blas
} // end namespace cusp

#include <cusp/blas/blas.inl>
