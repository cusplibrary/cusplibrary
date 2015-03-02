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

/*! \file smoothed_aggregation.h
 *  \brief Algebraic multigrid preconditoner based on smoothed aggregation.
 *
 */

#pragma once

#include <cusp/detail/config.h>

#include <cusp/relaxation/sor.h>

namespace cusp
{
namespace precond
{
namespace aggregation
{

/*! \addtogroup preconditioners Preconditioners
 *  \ingroup preconditioners
 *  \{
 */

template <typename ValueType, typename MemorySpace>
class sor_smoother
{
public:
    size_t num_iters;
    cusp::relaxation::sor<ValueType,MemorySpace> M;

    sor_smoother(void) {}

    template <typename ValueType2, typename MemorySpace2>
    gauss_seidel_smoother(const gauss_seidel_smoother<ValueType2,MemorySpace2>& A)
        : num_iters(A.num_iters), M(A.M) {}

    template <typename MatrixType1, typename MatrixType2>
    gauss_seidel_smoother(const MatrixType1& A, const sa_level<MatrixType2>& L)
    {
        initialize(A, L);
    }

    template <typename MatrixType1, typename MatrixType2>
    void initialize(const MatrixType1& A, const sa_level<MatrixType2>& L, ValueType omega=4.0/3.0)
    {
        M.default_omega = omega;

        M.temp.resize(A.num_rows);
        M.gs.ordering.resize(A.num_rows);
        M.gs.color_offsets.resize(A.num_rows);

        cusp::array1d<int,MemorySpace> colors(A.num_rows);
        int max_colors = cusp::graph::vertex_coloring(A, colors);

        thrust::sequence(M.gs.ordering.begin(), M.gs.ordering.end());
        thrust::sort_by_key(colors.begin(), colors.end(), M.gs.ordering.begin());

        cusp::array1d<int,MemorySpace> temp(max_colors + 1);
        thrust::reduce_by_key(colors.begin(),
                              colors.end(),
                              thrust::constant_iterator<int>(1),
                              thrust::make_discard_iterator(),
                              temp.begin());
        thrust::exclusive_scan(temp.begin(), temp.end(), temp.begin(), 0);
        M.gs.color_offsets = temp;

        cusp::extract_diagonal(A, M.gs.diagonal);
    }

    // smooths initial x
    template<typename MatrixType, typename VectorType1, typename VectorType2>
    void presmooth(const MatrixType& A, const VectorType1& b, VectorType2& x)
    {
        for(size_t i = 0; i < num_iters; i++)
            M(A, b, x);
    }

    // smooths initial x
    template<typename MatrixType, typename VectorType1, typename VectorType2>
    void postsmooth(const MatrixType& A, const VectorType1& b, VectorType2& x)
    {
        for(size_t i = 0; i < num_iters; i++)
            M(A, b, x);
    }
};
/*! \}
 */

} // end namespace aggregation
} // end namespace precond
} // end namespace cusp

