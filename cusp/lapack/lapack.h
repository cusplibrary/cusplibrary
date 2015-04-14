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

/*! \file lapack.h
 *  \brief Interface to lapack functions
 */

#pragma once

#include <cusp/detail/config.h>

#include <cusp/lapack/detail/defs.h>

namespace cusp
{
namespace lapack
{

/**
 * \brief Compute LU factorization of matrix
 *
 * \tparam Array2d Type of the input matrix to factor
 * \tparam Array1d Type of pivot array
 *
 * \param A Input matrix to factor
 * \param piv Array containing pivots
 *
 * \par Overview
 * This routine computes the LU factorization of a general m-by-n matrix A as A = P*L*U,
 * where P is a permutation matrix, L is lower triangular with unit diagonal elements
 * (lower trapezoidal if m > n) and U is upper triangular
 * (upper trapezoidal if m < n). The routine uses partial pivoting, with row interchanges.
 *
 * \par Example
 * \code
 * #include <cusp/array2d.h>
 * #include <cusp/print.h>
 *
 * #include <cusp/gallery/poisson.h>
 *
 * // include cusp lapack header file
 * #include <cusp/lapack/lapack.h>
 *
 * int main()
 * {
 *   // create an empty sparse matrix structure
 *   cusp::array2d<float,cusp::host_memory> A;
 *   cusp::array1d<float,cusp::host_memory> piv;
 *
 *   // create 2D Poisson problem
 *   cusp::gallery::poisson5pt(A, 4, 4);
 *
 *   // compute LU factorization of A
 *   cusp::lapack::getrf(A, piv);
 *
 *   // print the contents of A
 *   cusp::print(A);
 *   // print the contents of piv
 *   cusp::print(piv);
 * }
 * \endcode
 */
template<typename Array2d, typename Array1d>
void getrf( Array2d& A, Array1d& piv );

/**
 * \brief Computes the Cholesky factorization of a symmetric (Hermitian)
 * positive-definite matrix.
 *
 * \tparam Array2d Type of the input matrix to factor
 *
 * \param A Input matrix to factor
 *
 * \par Overview
 * This routine forms the Cholesky factorization of a symmetric positive-definite or,
 * for complex data, Hermitian positive-definite matrix A:
 * A = UT*U for real data, A = UH*U for complex data if uplo='U'
 * A = L*LT for real data, A = L*LH for complex data if uplo='L'
 * where L is a lower triangular matrix and U is upper triangular.
 *
 * \par Example
 * \code
 * #include <cusp/array2d.h>
 * #include <cusp/print.h>
 *
 * #include <cusp/gallery/poisson.h>
 *
 * // include cusp lapack header file
 * #include <cusp/lapack/lapack.h>
 *
 * int main()
 * {
 *   // create an empty sparse matrix structure
 *   cusp::array2d<float,cusp::host_memory> A;
 *
 *   // create 2D Poisson problem
 *   cusp::gallery::poisson5pt(A, 4, 4);
 *
 *   // compute Cholesky factorization of A
 *   cusp::lapack::potrf(A);
 *
 *   // print the contents of A
 *   cusp::print(A);
 * }
 * \endcode
 */
template<typename Array2d>
void potrf( Array2d& A, char uplo = 'U' );

/**
 * \brief Computes the Bunch-Kaufman factorization of a symmetric matrix.
 *
 * \tparam Array2d Type of the input matrix to factor
 * \tparam Array1d Type of pivot array
 *
 * \param A Input matrix to factor
 * \param piv Array containing pivots
 *
 * \par Overview
 * This routine computes the factorization of a symmetric or hermitian matrix
 * using the Bunch-Kaufman diagonal pivoting method. The form of the factorization is:
 * A = P*U*D*UT*PT, if uplo='U'
 * A = P*L*D*LT*PT, if uplo='L'
 * where A is the input matrix, P is a permutation matrix, U and L are upper and lower
 * triangular matrices with unit diagonal, and D is a symmetric block-diagonal matrix
 * with 1-by-1 and 2-by-2 diagonal blocks. U and L have 2-by-2 unit diagonal blocks
 * corresponding to the 2-by-2 blocks of D.
 *
 * \par Example
 * \code
 * #include <cusp/array1d.h>
 * #include <cusp/array2d.h>
 * #include <cusp/print.h>
 *
 * #include <cusp/gallery/poisson.h>
 *
 * // include cusp lapack header file
 * #include <cusp/lapack/lapack.h>
 *
 * int main()
 * {
 *   // create an empty sparse matrix structure
 *   cusp::array2d<float,cusp::host_memory> A;
 *   cusp::array1d<float,cusp::host_memory> piv;
 *
 *   // create 2D Poisson problem
 *   cusp::gallery::poisson5pt(A, 4, 4);
 *
 *   // compute Bunch-Kaufman factorization of A
 *   cusp::lapack::sytrf(A, piv);
 *
 *   // print the contents of A
 *   cusp::print(A);
 *   // print the contents of piv
 *   cusp::print(piv);
 * }
 * \endcode
 */
template<typename Array2d, typename Array1d>
void sytrf( Array2d& A, Array1d& piv, char uplo = 'U' );

/**
 * \brief Solves a system of linear equations with an LU-factored
 * square matrix, with multiple right-hand sides.
 *
 * \tparam Array2d Type of the input matrices
 * \tparam Array1d Type of pivot array
 *
 * \param A LU factored input matrix
 * \param piv Array containing pivots
 * \param B matrix containing multiple right-hand side vectors
 *
 * \par Overview
 * This routine solves for X the following systems of linear equations:
 * A*X = B if trans='N',
 * AT*X = B if trans='T',
 * AH*X = B if trans='C' (for complex matrices only).
 *
 * \note Before calling this routine, you must call getrf to compute the LU factorization of A.
 *
 * \par Example
 * \code
 * #include <cusp/array1d.h>
 * #include <cusp/array2d.h>
 * #include <cusp/print.h>
 *
 * #include <cusp/gallery/poisson.h>
 *
 * // include cusp lapack header file
 * #include <cusp/lapack/lapack.h>
 *
 * int main()
 * {
 *   // create an empty sparse matrix structure
 *   cusp::array2d<float,cusp::host_memory> A;
 *   cusp::array1d<float,cusp::host_memory> piv;
 *
 *   // create 2D Poisson problem
 *   cusp::gallery::poisson5pt(A, 4, 4);
 *
 *   // create initial RHS of 2 vectors and initialize
 *   cusp::array2d<float,cusp::host_memory> B(A.num_rows, 2);
 *   B.values = cusp::random_array<float>(B.values.size());
 *
 *   // compute LU factorization of A
 *   cusp::lapack::getrf(A, piv);
 *   // solve multiple RHS vectors
 *   cusp::lapack::getrs(A, piv, B);
 *
 *   // print the contents of B
 *   cusp::print(B);
 * }
 * \endcode
 */
template<typename Array2d, typename Array1d>
void getrs( const Array2d& A, const Array1d& piv, Array2d& B, char trans = 'N' );

template<typename Array2d>
void potrs( const Array2d& A, Array2d& B, char uplo = 'U');

template<typename Array2d, typename Array1d>
void sytrs( const Array2d& A, const Array1d& piv, Array2d& B, char uplo = 'U' );

/**
 * \brief Solves a system of linear equations with a triangular matrix,
 * with multiple right-hand sides.
 *
 * \tparam Array2d Type of the input matrices
 *
 * \param A Triangular input matrix
 * \param B matrix containing multiple right-hand side vectors
 * \param uplo Indicates whether A is upper or lower triangular
 * \param trans If 'N', then A*X = B is solved for X.
 * \param diag If 'N', then A is not a unit triangular matrix.
 *
 * \par Overview
 * This routine solves for X the following systems of linear equations with
 * a triangular matrix A, with multiple right-hand sides stored in B:
 * A*X = B if trans='N',
 * AT*X = B if trans='T',
 * AH*X = B if trans='C' (for complex matrices only).
 *
 * \par Example
 * \code
 * #include <cusp/array1d.h>
 * #include <cusp/array2d.h>
 * #include <cusp/print.h>
 *
 * #include <cusp/gallery/poisson.h>
 *
 * // include cusp lapack header file
 * #include <cusp/lapack/lapack.h>
 *
 * int main()
 * {
 *   // create an empty sparse matrix structure
 *   cusp::array2d<float,cusp::host_memory> A;
 *
 *   // create 2D Poisson problem
 *   cusp::gallery::poisson5pt(A, 4, 4);
 *
 *   // create initial RHS of 2 vectors and initialize
 *   cusp::array2d<float,cusp::host_memory> B(A.num_rows, 2);
 *   B.values = cusp::random_array<float>(B.values.size());
 *
 *   // solve multiple RHS vectors
 *   cusp::lapack::trtrs(A, B);
 *
 *   // print the contents of B
 *   cusp::print(B);
 * }
 * \endcode
 */
template<typename Array2d, typename Array1d>
void trtrs( const Array2d& A, Array2d& B, char uplo = 'U', char trans = 'N', char diag = 'N' );

template<typename Array2d>
void trtri( Array2d& A );

template<typename Array2d, typename Array1d>
void syev( const Array2d& A, Array1d& eigvals, Array2d& eigvecs );

template<typename Array1d1, typename Array1d2, typename Array1d3, typename Array2d>
void stev( const Array1d1& alphas, const Array1d2& betas,
           Array1d3& eigvals, Array2d& eigvecs, char job = EvalsOrEvecs<evecs>::type );

template<typename Array1d1, typename Array1d2, typename Array1d3>
void stev( const Array1d1& alphas, const Array1d2& betas, Array1d3& eigvals );

template<typename Array2d1, typename Array2d2, typename Array1d, typename Array2d3>
void sygv( const Array2d1& A, const Array2d2& B, Array1d& eigvals, Array2d3& eigvecs );

template<typename Array2d, typename Array1d>
void gesv( const Array2d& A, Array2d& B, Array1d& pivots );

} // end namespace lapack
} // end namespace cusp

#include <cusp/lapack/detail/lapack.inl>
