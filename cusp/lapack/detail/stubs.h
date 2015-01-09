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
 *  \brief Two-dimensional array
 */

#pragma once

namespace cusp
{
namespace lapack
{
namespace detail
{

lapack_int potrf( lapack_int order, char uplo, lapack_int n, float* a, lapack_int lda)
{
    return LAPACKE_spotrf(order, uplo, n, a, lda);
}
lapack_int potrf( lapack_int order, char uplo, lapack_int n, double* a, lapack_int lda)
{
    return LAPACKE_dpotrf(order, uplo, n, a, lda);
}

lapack_int trtri( lapack_int order, char uplo, char diag, lapack_int n, float* a, lapack_int lda)
{
    return LAPACKE_strtri(order, uplo, diag, n, a, lda);
}
lapack_int trtri( lapack_int order, char uplo, char diag, lapack_int n, double* a, lapack_int lda)
{
    return LAPACKE_dtrtri(order, uplo, diag, n, a, lda);
}

lapack_int syev( lapack_int order, char job, char uplo, lapack_int n, float* a, lapack_int lda, float* w )
{
    return LAPACKE_ssyev(order, job, uplo, n, a, lda, w);
}
lapack_int syev( lapack_int order, char job, char uplo, lapack_int n, double* a, lapack_int lda, double* w )
{
    return LAPACKE_dsyev(order, job, uplo, n, a, lda, w);
}

lapack_int stev( lapack_int order, char job, lapack_int n, float* a, float* b, float* z, lapack_int ldz)
{
    return LAPACKE_sstev(order, job, n, a, b, z, ldz);
}
lapack_int stev( lapack_int order, char job, lapack_int n, double* a, double* b, double* z, lapack_int ldz)
{
    return LAPACKE_dstev(order, job, n, a, b, z, ldz);
}

lapack_int sygv( lapack_int order, lapack_int itype, char job, char uplo, lapack_int n,
                 float* a, lapack_int lda, float* b, lapack_int ldb, float* w )
{
    return LAPACKE_ssygv(order, itype, job, uplo, n, a, lda, b, ldb, w);
}
lapack_int sygv( lapack_int order, lapack_int itype, char job, char uplo, lapack_int n,
                 double* a, lapack_int lda, double* b, lapack_int ldb, double* w )
{
    return LAPACKE_dsygv(order, itype, job, uplo, n, a, lda, b, ldb, w);
}

lapack_int gesv( lapack_int order, lapack_int n, lapack_int nrhs,
                  float* a, lapack_int lda, lapack_int* ipiv, float* b, lapack_int ldb )
{
    return LAPACKE_sgesv(order, n, nrhs, a, lda, ipiv, b, ldb);
}
lapack_int gesv( lapack_int order, lapack_int n, lapack_int nrhs,
                  double* a, lapack_int lda, lapack_int* ipiv, double* b, lapack_int ldb )
{
    return LAPACKE_dgesv(order, n, nrhs, a, lda, ipiv, b, ldb);
}

} // end namespace detail
} // end namespace lapack
} // end namespace cusp
