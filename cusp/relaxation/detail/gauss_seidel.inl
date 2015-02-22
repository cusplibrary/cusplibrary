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

/*! \file gauss_seidel.inl
 *  \brief Inline file for gauss_seidel.h
 */

#include <cusp/multiply.h>
#include <cusp/format_utils.h>
#include <cusp/graph/vertex_coloring.h>

#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

namespace cusp
{
namespace relaxation
{

template <typename ValueType, typename MemorySpace>
template<typename MatrixType>
gauss_seidel<ValueType,MemorySpace>
::gauss_seidel(const MatrixType& A, sweep default_direction)
    : ordering(A.num_rows), default_direction(default_direction)
{
    cusp::array1d<int,MemorySpace> colors(A.num_rows);
    int max_colors = cusp::graph::vertex_coloring(A, colors);

    thrust::sequence(ordering.begin(), ordering.end());
    thrust::sort_by_key(colors.begin(), colors.end(), ordering.begin());

    cusp::array1d<int,MemorySpace> temp(max_colors + 1);
    thrust::reduce_by_key(colors.begin(),
                          colors.end(),
                          thrust::constant_iterator<int>(1),
                          thrust::make_discard_iterator(),
                          temp.begin());
    thrust::exclusive_scan(temp.begin(), temp.end(), temp.begin(), 0);
    color_offsets = temp;
}

template <typename ValueType, typename MemorySpace>
template<typename MatrixType>
gauss_seidel<ValueType,MemorySpace>
::gauss_seidel(const cusp::precond::aggregation::sa_level<MatrixType>& sa_level, sweep default_direction)
    : ordering(sa_level.A_.num_rows), default_direction(default_direction)
{
    const MatrixType& A = sa_level.A_;

    cusp::array1d<int,MemorySpace> colors(A.num_rows);
    int max_colors = cusp::graph::vertex_coloring(A, colors);

    thrust::sequence(ordering.begin(), ordering.end());
    thrust::sort_by_key(colors.begin(), colors.end(), ordering.begin());

    cusp::array1d<int,MemorySpace> temp(max_colors + 1);
    thrust::reduce_by_key(colors.begin(),
                          colors.end(),
                          thrust::constant_iterator<int>(1),
                          thrust::make_discard_iterator(),
                          temp.begin());
    thrust::exclusive_scan(temp.begin(), temp.end(), temp.begin(), 0);
    color_offsets = temp;
}

// linear_operator
template <typename ValueType, typename MemorySpace>
template<typename MatrixType, typename VectorType1, typename VectorType2>
void gauss_seidel<ValueType,MemorySpace>
::operator()(const MatrixType& A, const VectorType1& b, VectorType2& x)
{
    gauss_seidel<ValueType,MemorySpace>::operator()(A,b,x,default_direction);
}

// override default sweep direction
template <typename ValueType, typename MemorySpace>
template<typename MatrixType, typename VectorType1, typename VectorType2>
void gauss_seidel<ValueType,MemorySpace>
::operator()(const MatrixType& A, const VectorType1& b, VectorType2& x, sweep direction)
{
    using thrust::system::detail::generic::select_system;

    MemorySpace system;

    if(direction == FORWARD)
    {
        for(int i = 0; i < color_offsets.size()-1; i++)
            gauss_seidel_indexed(thrust::detail::derived_cast(system),
                A, x, b, ordering, color_offsets[i], color_offsets[i+1], 1);
    }
    else if(direction == BACKWARD)
    {
        for(int i = color_offsets.size()-1; i > 0; i--)
            gauss_seidel_indexed(thrust::detail::derived_cast(system),
                A, x, b, ordering, color_offsets[i-1], color_offsets[i], 1);
    }
    else if(direction == SYMMETRIC)
    {
        operator()(A, b, x, FORWARD);
        operator()(A, b, x, BACKWARD);
    }
    else
    {
        throw cusp::runtime_exception("Unknown Gauss-Seidel sweep direction specified.");
    }
}

template <typename ValueType, typename MemorySpace>
template<typename MatrixType, typename VectorType1, typename VectorType2>
void gauss_seidel<ValueType,MemorySpace>
::presmooth(const MatrixType& A, const VectorType1& b, VectorType2& x)
{
    // TODO : specialize this using initial x of all 0s
    gauss_seidel<ValueType,MemorySpace>::operator()(A,b,x,default_direction);
}

template <typename ValueType, typename MemorySpace>
template<typename MatrixType, typename VectorType1, typename VectorType2>
void gauss_seidel<ValueType,MemorySpace>
::postsmooth(const MatrixType& A, const VectorType1& b, VectorType2& x)
{
    gauss_seidel<ValueType,MemorySpace>::operator()(A,b,x,default_direction);
}

} // end namespace relaxation
} // end namespace cusp

