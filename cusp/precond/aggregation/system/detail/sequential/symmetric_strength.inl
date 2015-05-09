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


#include <cusp/array1d.h>
#include <cusp/convert.h>
#include <cusp/copy.h>
#include <cusp/csr_matrix.h>
#include <cusp/format_utils.h>

#include <cusp/detail/functional.h>

#include <thrust/count.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/permutation_iterator.h>

namespace cusp
{
namespace precond
{
namespace aggregation
{
namespace detail
{

template <typename DerivedPolicy, typename MatrixType1, typename MatrixType2>
void symmetric_strength_of_connection(sequential::execution_policy<DerivedPolicy> &exec,
                                      const MatrixType1& A, MatrixType2& S, const double theta)
{
    typedef typename MatrixType1::index_type   IndexType;
    typedef typename MatrixType1::value_type   ValueType;
    typedef typename MatrixType1::memory_space MemorySpace;

    // extract matrix diagonal
    cusp::array1d<ValueType,MemorySpace> diagonal;
    cusp::extract_diagonal(exec, A, diagonal);

    IndexType num_entries = 0;

    // count num_entries in output
    for(size_t i = 0; i < A.num_rows; i++)
    {
        const ValueType Aii = diagonal[i];

        for(IndexType jj = A.row_offsets[i]; jj < A.row_offsets[i + 1]; jj++)
        {
            const IndexType   j = A.column_indices[jj];
            const ValueType Aij = A.values[jj];
            const ValueType Ajj = diagonal[j];

            //  |A(i,j)| >= theta * sqrt(|A(i,i)|*|A(j,j)|)
            if(Aij*Aij >= (theta * theta) * cusp::detail::absolute_value()(Aii * Ajj))
                num_entries++;
        }
    }

    // resize output
    S.resize(A.num_rows, A.num_cols, num_entries);

    // reset counter for second pass
    num_entries = 0;

    // copy strong connections to output
    for(size_t i = 0; i < A.num_rows; i++)
    {
        const ValueType Aii = diagonal[i];

        S.row_offsets[i] = num_entries;

        for(IndexType jj = A.row_offsets[i]; jj < A.row_offsets[i + 1]; jj++)
        {
            const IndexType   j = A.column_indices[jj];
            const ValueType Aij = A.values[jj];
            const ValueType Ajj = diagonal[j];

            //  |A(i,j)| >= theta * sqrt(|A(i,i)|*|A(j,j)|)
            if(Aij*Aij >= (theta * theta) * cusp::detail::absolute_value()(Aii * Ajj))
            {
                S.column_indices[num_entries] =   j;
                S.values[num_entries]         = Aij;
                num_entries++;
            }
        }
    }

    S.row_offsets[S.num_rows] = num_entries;
}

} // end namespace detail
} // end namespace aggregation
} // end namespace precond
} // end namespace cusp

