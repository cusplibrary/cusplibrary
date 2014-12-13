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
#include <cusp/precond/aggregation/smoothed_aggregation_options.h>

#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

namespace cusp
{
namespace relaxation
{
namespace detail
{
template<typename MatrixType, typename ArrayType1, typename ArrayType2>
void gauss_seidel_indexed(const MatrixType& A,
                          ArrayType1&  x,
                          const ArrayType1&  b,
                          const ArrayType2& indices,
                          const int row_start,
                          const int row_stop,
                          const int row_step)
{
    typedef typename ArrayType1::value_type V;
    typedef typename ArrayType2::value_type I;

    for(int i = row_start; i != row_stop; i += row_step)
    {
        I inew  = indices[i];
        I start = A.row_offsets[inew];
        I end   = A.row_offsets[inew + 1];
        V rsum  = 0;
        V diag  = 0;

        for(I jj = start; jj < end; ++jj)
        {
            I j = A.column_indices[jj];
            if (inew == j)
            {
                diag = A.values[jj];
            }
            else
            {
                rsum += A.values[jj]*x[j];
            }
        }

        if (diag != 0)
        {
            x[inew] = (b[inew] - rsum)/diag;
        }
    }
}
} // end namespace detail

template <typename ValueType, typename MemorySpace>
template<typename MatrixType>
gauss_seidel<ValueType,MemorySpace>
::gauss_seidel(const MatrixType& A, sweep default_direction)
    : ordering(A.num_rows), default_direction(default_direction)
{
    cusp::array1d<int,MemorySpace> colors(A.num_rows);
    int max_colors = cusp::graph::vertex_coloring(A, colors);

    color_offsets.resize(max_colors + 1);
    thrust::sequence(ordering.begin(), ordering.end());

    thrust::sort_by_key(colors.begin(), colors.end(), ordering.begin());
    thrust::reduce_by_key(colors.begin(),
                          colors.end(),
                          thrust::constant_iterator<int>(1),
                          thrust::make_discard_iterator(),
                          color_offsets.begin());
    thrust::exclusive_scan(color_offsets.begin(), color_offsets.end(), color_offsets.begin(), 0);
}

template <typename ValueType, typename MemorySpace>
template<typename MatrixType>
gauss_seidel<ValueType,MemorySpace>
::gauss_seidel(const cusp::precond::aggregation::sa_level<MatrixType>& sa_level)
    : ordering(sa_level.A.num_rows)
{
}

// linear_operator
template <typename ValueType, typename MemorySpace>
template<typename MatrixType, typename VectorType1, typename VectorType2>
void gauss_seidel<ValueType,MemorySpace>
::operator()(const MatrixType& A, const VectorType1& b, VectorType2& x)
{
    gauss_seidel<ValueType,MemorySpace>::operator()(A,b,x,default_direction);
}

// override default omega
template <typename ValueType, typename MemorySpace>
template<typename MatrixType, typename VectorType1, typename VectorType2>
void gauss_seidel<ValueType,MemorySpace>
::operator()(const MatrixType& A, const VectorType1& b, VectorType2& x, sweep direction)
{
    if(direction == FORWARD)
    {
        for(int i = 0; i < color_offsets.size()-1; i++)
            detail::gauss_seidel_indexed(A, x, b, ordering, color_offsets[i], color_offsets[i+1], 1);
    }
    else if(direction == BACKWARD)
    {
        for(int i = color_offsets.size()-1; i > 0; i--)
            detail::gauss_seidel_indexed(A, x, b, ordering, color_offsets[i-1], color_offsets[i], 1);
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
::presmooth(const MatrixType&, const VectorType1& b, VectorType2& x)
{
}

template <typename ValueType, typename MemorySpace>
template<typename MatrixType, typename VectorType1, typename VectorType2>
void gauss_seidel<ValueType,MemorySpace>
::postsmooth(const MatrixType& A, const VectorType1& b, VectorType2& x)
{
}

} // end namespace relaxation
} // end namespace cusp

