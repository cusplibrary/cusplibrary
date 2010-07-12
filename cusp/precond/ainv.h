/*
 *  Copyright 2008-2010 NVIDIA Corporation
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

/*! \file ainv.h
 *  \brief Approximate Inverse (AINV) preconditioner.
 */

#pragma once

#include <cusp/detail/config.h>

#include <cusp/linear_operator.h>
#include <cusp/hyb_matrix.h>

namespace cusp
{
namespace precond
{

/*! \addtogroup preconditioners Preconditioners
 *  \ingroup preconditioners
 *  \{
 */

/*! \p scaled_bridson_ainv : Approximate Inverse preconditoner (from Bridson's "outer product" formulation)
 *  The diagonal matrix is folded into the factorization to reduce operation count during 
 *  preconditioner application.  Not sure if this is a good idea or not, yet.
 *  This preconditioner allows for a novel dropping strategy, where rather than a fixed
 *  drop tolerance, you can specify now many non-zeroes are allowed per row.  The non-zeroes
 *  will be chosen based on largest magnitude.  This idea has been applied to IC factorization,
 *  but not AINV as far as I'm aware.  See:
 *  Lin, C. and More, J. J. 1999. Incomplete Cholesky Factorizations with Limited Memory. 
 *  SIAM J. Sci. Comput. 21, 1 (Aug. 1999), 24-45. 
 *  This preconditioner will only work for SPD matrices. 
 *
 */
template <typename ValueType, typename MemorySpace>
class scaled_bridson_ainv : public linear_operator<ValueType, MemorySpace>
{       
    typedef linear_operator<ValueType, MemorySpace> Parent;

public:
    cusp::hyb_matrix<int, ValueType, MemorySpace> w;
    cusp::hyb_matrix<int, ValueType, MemorySpace> w_t;

    /*! construct a \p ainv preconditioner
     *
     * \param A matrix to precondition
     * \tparam MatrixType matrix
     * \param ValueType drop_tolerance Tolerance for dropping during factorization
     * \param sparsity_pattern Count of non-zeroes allowed per row of the factored matrix.  If negative, this will be ignored.
     */
    template<typename MatrixTypeA>
    scaled_bridson_ainv(const MatrixTypeA & A, ValueType drop_tolerance=0.1, int sparsity_pattern=-1);
        
    /*! apply the preconditioner to vector \p x and store the result in \p y
     *
     * \param x input vector
     * \param y ouput vector
     * \tparam VectorType1 vector
     * \tparam VectorType2 vector
     */
    template <typename VectorType1, typename VectorType2>
    void operator()(const VectorType1& x, VectorType2& y) const;
};
/*! \}
 */


/*! \addtogroup preconditioners Preconditioners
 *  \ingroup preconditioners
 *  \{
 */

/*! \p bridson_ainv : Approximate Inverse preconditoner (from Bridson's "outer product" formulation)
 *  This preconditioner allows for a novel dropping strategy, where rather than a fixed
 *  drop tolerance, you can specify now many non-zeroes are allowed per row.  The non-zeroes
 *  will be chosen based on largest magnitude.  This idea has been applied to IC factorization,
 *  but not AINV as far as I'm aware.  See:
 *  Lin, C. and More, J. J. 1999. Incomplete Cholesky Factorizations with Limited Memory. 
 *  SIAM J. Sci. Comput. 21, 1 (Aug. 1999), 24-45. 
 *  This preconditioner will only work for SPD matrices. 
 */

template <typename ValueType, typename MemorySpace>
class bridson_ainv : public linear_operator<ValueType, MemorySpace>
{       
    typedef linear_operator<ValueType, MemorySpace> Parent;

public:
    cusp::hyb_matrix<int, ValueType, MemorySpace> w;
    cusp::hyb_matrix<int, ValueType, MemorySpace> w_t;
    cusp::array1d<ValueType, MemorySpace> diagonals;

    /*! construct a \p ainv preconditioner
     *
     * \param A matrix to precondition
     * \tparam MatrixTypeA matrix
     * \param ValueType drop_tolerance Tolerance for dropping during factorization
     * \param sparsity_pattern Count of non-zeroes allowed per row of the factored matrix.  If negative, this will be ignored.
     */
    template<typename MatrixTypeA>
    bridson_ainv(const MatrixTypeA & A, ValueType drop_tolerance=0.1, int sparsity_pattern=-1);
        
    /*! apply the preconditioner to vector \p x and store the result in \p y
     *
     * \param x input vector
     * \param y ouput vector
     * \tparam VectorType1 vector
     * \tparam VectorType2 vector
     */
    template <typename VectorType1, typename VectorType2>
    void operator()(const VectorType1& x, VectorType2& y) const;
};
/*! \}
 */

} // end namespace precond
} // end namespace cusp

#include <cusp/precond/detail/ainv.inl>

