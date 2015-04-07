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

#pragma once

#include <cusp/detail/config.h>

#include <cusp/gallery/stencil.h>

namespace cusp
{
namespace gallery
{

template <typename MatrixType>
void poisson5pt(MatrixType& matrix, size_t m, size_t n)
{
    typedef typename MatrixType::index_type IndexType;
    typedef typename MatrixType::value_type ValueType;
    typedef thrust::tuple<IndexType,IndexType>    StencilIndex;
    typedef thrust::tuple<StencilIndex,ValueType> StencilPoint;

    cusp::array1d<StencilPoint, cusp::host_memory> stencil;
    stencil.push_back(StencilPoint(StencilIndex(  0, -1), ValueType(-1)));
    stencil.push_back(StencilPoint(StencilIndex( -1,  0), ValueType(-1)));
    stencil.push_back(StencilPoint(StencilIndex(  0,  0), ValueType( 4)));
    stencil.push_back(StencilPoint(StencilIndex(  1,  0), ValueType(-1)));
    stencil.push_back(StencilPoint(StencilIndex(  0,  1), ValueType(-1)));

    cusp::gallery::generate_matrix_from_stencil(matrix, stencil, StencilIndex(m,n));
}

template <typename MatrixType>
void poisson9pt(MatrixType& matrix, size_t m, size_t n)
{
    typedef typename MatrixType::index_type IndexType;
    typedef typename MatrixType::value_type ValueType;
    typedef thrust::tuple<IndexType,IndexType>    StencilIndex;
    typedef thrust::tuple<StencilIndex,ValueType> StencilPoint;

    cusp::array1d<StencilPoint, cusp::host_memory> stencil;
    stencil.push_back(StencilPoint(StencilIndex( -1, -1), ValueType(-1)));
    stencil.push_back(StencilPoint(StencilIndex(  0, -1), ValueType(-1)));
    stencil.push_back(StencilPoint(StencilIndex(  1, -1), ValueType(-1)));
    stencil.push_back(StencilPoint(StencilIndex( -1,  0), ValueType(-1)));
    stencil.push_back(StencilPoint(StencilIndex(  0,  0), ValueType( 8)));
    stencil.push_back(StencilPoint(StencilIndex(  1,  0), ValueType(-1)));
    stencil.push_back(StencilPoint(StencilIndex( -1,  1), ValueType(-1)));
    stencil.push_back(StencilPoint(StencilIndex(  0,  1), ValueType(-1)));
    stencil.push_back(StencilPoint(StencilIndex(  1,  1), ValueType(-1)));

    cusp::gallery::generate_matrix_from_stencil(matrix, stencil, StencilIndex(m,n));
}

template <typename MatrixType>
void poisson7pt(MatrixType& matrix, size_t m, size_t n, size_t k)
{
    typedef typename MatrixType::index_type IndexType;
    typedef typename MatrixType::value_type ValueType;
    typedef thrust::tuple<IndexType,IndexType,IndexType>    StencilIndex;
    typedef thrust::tuple<StencilIndex,ValueType> 	    StencilPoint;

    cusp::array1d<StencilPoint, cusp::host_memory> stencil;
    stencil.push_back(StencilPoint(StencilIndex( 0,  0, -1), ValueType(-1)));
    stencil.push_back(StencilPoint(StencilIndex( 0, -1,  0), ValueType(-1)));
    stencil.push_back(StencilPoint(StencilIndex(-1,  0,  0), ValueType(-1)));
    stencil.push_back(StencilPoint(StencilIndex( 0,  0,  0), ValueType( 6)));
    stencil.push_back(StencilPoint(StencilIndex( 1,  0,  0), ValueType(-1)));
    stencil.push_back(StencilPoint(StencilIndex( 0,  1,  0), ValueType(-1)));
    stencil.push_back(StencilPoint(StencilIndex( 0,  0,  1), ValueType(-1)));

    cusp::gallery::generate_matrix_from_stencil(matrix, stencil, StencilIndex(m,n,k));
}

template <typename MatrixType>
void poisson27pt(MatrixType& matrix, size_t m, size_t n, size_t l)
{
    typedef typename MatrixType::index_type IndexType;
    typedef typename MatrixType::value_type ValueType;
    typedef thrust::tuple<IndexType,IndexType,IndexType>    StencilIndex;
    typedef thrust::tuple<StencilIndex,ValueType> 	    StencilPoint;

    cusp::array1d<StencilPoint, cusp::host_memory> stencil;
    for( IndexType k = -1; k <= 1; k++ )
        for( IndexType j = -1; j <= 1; j++ )
            for( IndexType i = -1; i <= 1; i++ )
                stencil.push_back(StencilPoint(StencilIndex( i, j, k), (i==0 && j==0 && k==0) ? ValueType(26) : ValueType(-1)));

    cusp::gallery::generate_matrix_from_stencil(matrix, stencil, StencilIndex(m,n,l));
}

} // end namespace gallery
} // end namespace cusp

