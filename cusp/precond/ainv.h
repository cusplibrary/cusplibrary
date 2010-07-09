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

/*! \p ainv : Approximate Inverse preconditoner (from Bridson's "outer product" formulation)
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
     */
    template<typename MatrixTypeA>
    scaled_bridson_ainv(const MatrixTypeA & A, ValueType drop_tolerance=0.1);
        
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
     * \tparam MatrixType matrix
     */
    template<typename MatrixTypeA>
    bridson_ainv(const MatrixTypeA & A, ValueType drop_tolerance=0.1);
        
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

