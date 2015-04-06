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

#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/elementwise.h>
#include <cusp/sort.h>

#include <cusp/blas/blas.h>
#include <cusp/eigen/spectral_radius.h>

#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <thrust/gather.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>

#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

namespace cusp
{
namespace precond
{
namespace aggregation
{
namespace detail
{

using namespace thrust::placeholders;

template <typename MatrixType, typename ValueType>
void smooth_prolongator(const MatrixType& S,
                        const MatrixType& T,
                        MatrixType& P,
                        const ValueType omega,
                        const ValueType rho_Dinv_S,
                        cusp::coo_format,
                        cusp::device_memory)
{
    typedef typename MatrixType::index_type IndexType;

    // TODO handle case with unaggregated nodes more gracefully
    if (T.num_entries == T.num_rows) {

        const ValueType lambda = omega / rho_Dinv_S;

        // temp <- -lambda * S(i,j) * T(j,k)
        MatrixType temp(S.num_rows, T.num_cols, S.num_entries + T.num_entries);
        thrust::copy(S.row_indices.begin(), S.row_indices.end(), temp.row_indices.begin());
        thrust::gather(S.column_indices.begin(), S.column_indices.end(), T.column_indices.begin(), temp.column_indices.begin());
        thrust::transform(S.values.begin(), S.values.end(),
                          thrust::make_permutation_iterator(T.values.begin(), S.column_indices.begin()),
                          temp.values.begin(),
                          -lambda * _1 * _2);

        // temp <- D^-1
        {
            cusp::array1d<ValueType, cusp::device_memory> D(S.num_rows);
            cusp::extract_diagonal(S, D);
            thrust::transform(temp.values.begin(), temp.values.begin() + S.num_entries,
                              thrust::make_permutation_iterator(D.begin(), S.row_indices.begin()),
                              temp.values.begin(),
                              thrust::divides<ValueType>());
        }

        // temp <- temp + T
        thrust::copy(T.row_indices.begin(),    T.row_indices.end(),    temp.row_indices.begin()    + S.num_entries);
        thrust::copy(T.column_indices.begin(), T.column_indices.end(), temp.column_indices.begin() + S.num_entries);
        thrust::copy(T.values.begin(),         T.values.end(),         temp.values.begin()         + S.num_entries);

        // sort by (I,J)
        cusp::sort_by_row_and_column(temp.row_indices, temp.column_indices, temp.values, 0, T.num_rows, 0, T.num_cols);

        // compute unique number of nonzeros in the output
        // throws a warning at compile (warning: expression has no effect)
        IndexType NNZ = thrust::inner_product(thrust::make_zip_iterator(thrust::make_tuple(temp.row_indices.begin(), temp.column_indices.begin())),
                                              thrust::make_zip_iterator(thrust::make_tuple(temp.row_indices.end (),  temp.column_indices.end()))   - 1,
                                              thrust::make_zip_iterator(thrust::make_tuple(temp.row_indices.begin(), temp.column_indices.begin())) + 1,
                                              IndexType(0),
                                              thrust::plus<IndexType>(),
                                              thrust::not_equal_to< thrust::tuple<IndexType,IndexType> >()) + 1;

        // allocate space for output
        P.resize(temp.num_rows, temp.num_cols, NNZ);

        // sum values with the same (i,j)
        thrust::reduce_by_key(thrust::make_zip_iterator(thrust::make_tuple(temp.row_indices.begin(), temp.column_indices.begin())),
                              thrust::make_zip_iterator(thrust::make_tuple(temp.row_indices.end(),   temp.column_indices.end())),
                              temp.values.begin(),
                              thrust::make_zip_iterator(thrust::make_tuple(P.row_indices.begin(), P.column_indices.begin())),
                              P.values.begin(),
                              thrust::equal_to< thrust::tuple<IndexType,IndexType> >(),
                              thrust::plus<ValueType>());

    } else {

        cusp::array1d<ValueType, cusp::device_memory> D(S.num_rows);
        cusp::extract_diagonal(S, D);

        // create D_inv_S by copying S then scaling
        MatrixType D_inv_S(S);
        // scale the rows of D_inv_S by D^-1
        thrust::transform(D_inv_S.values.begin(), D_inv_S.values.begin() + S.num_entries,
                          thrust::make_permutation_iterator(D.begin(), S.row_indices.begin()),
                          D_inv_S.values.begin(),
                          thrust::divides<ValueType>());

        const ValueType lambda = omega / rho_Dinv_S;
        cusp::blas::scal( D_inv_S.values, lambda );

        MatrixType temp;
        cusp::multiply( D_inv_S, T, temp );
        cusp::subtract( T, temp, P );

    }
}

template <typename MatrixType, typename ValueType>
void smooth_prolongator(const MatrixType& S,
                        const MatrixType& T,
                        MatrixType& P,
                        const ValueType omega,
                        const ValueType rho_Dinv_S,
                        cusp::csr_format,
                        cusp::host_memory)
{
    typedef typename MatrixType::index_type IndexType;

    cusp::array1d<ValueType, cusp::host_memory> D(S.num_rows);
    cusp::extract_diagonal(S, D);

    // create D_inv_S by copying S then scaling
    MatrixType D_inv_S(S);
    // scale the rows of D_inv_S by D^-1
    for ( size_t row = 0; row < D_inv_S.num_rows; row++ )
    {
        const IndexType row_start = D_inv_S.row_offsets[row];
        const IndexType row_end   = D_inv_S.row_offsets[row+1];
        const ValueType diagonal  = D[row];

        for ( IndexType index = row_start; index < row_end; index++ )
            D_inv_S.values[index] /= diagonal;
    }

    const ValueType lambda = omega / rho_Dinv_S;
    cusp::blas::scal( D_inv_S.values, lambda );

    MatrixType temp;
    cusp::multiply( D_inv_S, T, temp );
    cusp::subtract( T, temp, P );
}

} // end namespace detail

template <typename MatrixType, typename ValueType>
void smooth_prolongator(const MatrixType& S,
                        const MatrixType& T,
                        MatrixType& P,
                        const ValueType omega,
                        const ValueType rho_Dinv_S)
{
    detail::smooth_prolongator(S, T, P, omega, rho_Dinv_S, typename MatrixType::format(), typename MatrixType::memory_space());
}

} // end namespace aggregation
} // end namespace precond
} // end namespace cusp

