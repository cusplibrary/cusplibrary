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

/*! \file lapack.inl
 *  \brief Definition of lapack interface routines
 */

#include <cusp/array1d.h>
#include <cusp/array2d.h>

#include <cusp/lapack/detail/stubs.h>

namespace cusp
{
namespace lapack
{

template<typename Array2d>
void potrf( Array2d& A )
{
    typedef typename Array2d::value_type ValueType;

    lapack_int order = Orientation<typename Array2d::orientation>::type;
    char uplo = UpperOrLower<upper>::type;

    lapack_int n = A.num_rows;
    lapack_int lda = A.pitch;
    ValueType *a = thrust::raw_pointer_cast(&A(0,0));
    lapack_int info = detail::potrf(order, uplo, n, a, lda);

    if( info != 0 )
        throw cusp::runtime_exception("potrf failed");
}

template<typename Array2d>
void trtri( Array2d& A )
{
    typedef typename Array2d::value_type ValueType;

    lapack_int order = Orientation<typename Array2d::orientation>::type;
    char uplo = UpperOrLower<upper>::type;
    char diag = UnitOrNonunit<nonunit>::type;

    lapack_int n = A.num_rows;
    lapack_int lda = A.pitch;
    ValueType *a = thrust::raw_pointer_cast(&A(0,0));
    lapack_int info = detail::trtri(order, uplo, diag, n, a, lda);

    if( info != 0 )
        throw cusp::runtime_exception("trtri failed");
}

template<typename Array2d, typename Array1d>
void syev( const Array2d& A, Array1d& eigvals, Array2d& eigvecs )
{
    typedef typename Array2d::value_type ValueType;

    eigvecs = A;

    lapack_int order = Orientation<typename Array2d::orientation>::type;
    char job  = EvalsOrEvecs<evecs>::type;
    char uplo = UpperOrLower<upper>::type;

    lapack_int n = A.num_rows;
    lapack_int lda = A.pitch;
    ValueType *a = thrust::raw_pointer_cast(&eigvecs(0,0));
    ValueType *w = thrust::raw_pointer_cast(&eigvals[0]);
    lapack_int info = detail::syev(order, job, uplo, n, a, lda, w);

    if( info != 0 )
        throw cusp::runtime_exception("syev failed");
}

template<typename Array1d1, typename Array1d2, typename Array1d3, typename Array2d>
void stev( const Array1d1& alphas, const Array1d2& betas, Array1d3& eigvals, Array2d& eigvecs, char job )
{
    typedef typename Array2d::value_type ValueType;

    cusp::array1d<ValueType,cusp::host_memory> temp(betas);
    eigvals = alphas;

    lapack_int order = Orientation<typename Array2d::orientation>::type;
    lapack_int n = alphas.size();
    lapack_int ldz = n;
    ValueType *a = thrust::raw_pointer_cast(&eigvals[0]);
    ValueType *b = thrust::raw_pointer_cast(&temp[0]);
    ValueType *z = thrust::raw_pointer_cast(&eigvecs(0,0));

    lapack_int info = detail::stev(order, job, n, a, b, z, ldz);

    if( info != 0 )
    {
        printf("stev failure code : %d\n", info);
        throw cusp::runtime_exception("stev failed");
    }
}

template<typename Array1d1, typename Array1d2, typename Array1d3>
void stev( const Array1d1& alphas, const Array1d2& betas, Array1d3& eigvals )
{
  typedef typename Array1d1::value_type ValueType;

  char job  = EvalsOrEvecs<evals>::type;
  cusp::array2d<ValueType,cusp::host_memory> eigvecs;

  stev(alphas, betas, eigvals, eigvecs, job);
}

template<typename Array2d1, typename Array2d2, typename Array1d, typename Array2d3>
void sygv( const Array2d1& A, const Array2d2& B, Array1d& eigvals, Array2d3& eigvecs )
{
    typedef typename Array2d1::value_type ValueType;
    typedef typename Array2d1::orientation Array2dOrientation;

    eigvecs = A;
    cusp::array2d<ValueType,cusp::host_memory,Array2dOrientation> temp(B);

    lapack_int order = Orientation<Array2dOrientation>::type;
    char itype = GenEigOp<gen_op1>::type;
    char job   = EvalsOrEvecs<evecs>::type;
    char uplo  = UpperOrLower<upper>::type;

    lapack_int n = A.num_rows;
    lapack_int lda = A.pitch;
    lapack_int ldb = B.pitch;
    ValueType *a = thrust::raw_pointer_cast(&eigvecs(0,0));
    ValueType *b = thrust::raw_pointer_cast(&temp(0,0));
    ValueType *w = thrust::raw_pointer_cast(&eigvals[0]);
    lapack_int info = detail::sygv(order, itype, job, uplo, n, a, lda, b, ldb, w);

    if( info != 0 )
    {
        printf("sygv failure code : %d\n", info);
        throw cusp::runtime_exception("sygv failed");
    }
}

template<typename Array2d, typename Array1d>
void gesv( const Array2d& A, Array2d& B, Array1d& pivots )
{
    typedef typename Array1d::value_type IndexType;
    typedef typename Array2d::value_type ValueType;

    Array2d C = A;

    lapack_int order = Orientation<typename Array2d::orientation>::type;

    lapack_int n = C.num_rows;
    lapack_int nrhs = B.num_cols;
    lapack_int ldc = C.pitch;
    lapack_int ldb = B.pitch;
    ValueType *c = thrust::raw_pointer_cast(&C(0,0));
    ValueType *b = thrust::raw_pointer_cast(&B(0,0));
    IndexType *ipiv = thrust::raw_pointer_cast(&pivots[0]);
    lapack_int info = detail::gesv(order, n, nrhs, c, ldc, ipiv, b, ldb);

    if( info != 0 )
        throw cusp::runtime_exception("gesv failed");
}

} // end namespace lapack
} // end namespace cusp
