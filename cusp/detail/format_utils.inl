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
    
#include <cusp/detail/matrix_traits.h>

#include <cusp/array1d.h>

#include <thrust/fill.h>
#include <thrust/extrema.h>
#include <thrust/binary_search.h>
#include <thrust/transform.h>
#include <thrust/scatter.h>

namespace cusp
{
namespace detail
{

template <typename OffsetArray, typename IndexArray>
void offsets_to_indices(const OffsetArray& offsets, IndexArray& indices)
{
    typedef typename OffsetArray::value_type OffsetType;

    // convert compressed row offsets into uncompressed row indices
    thrust::upper_bound(offsets.begin() + 1,
                        offsets.end(),
                        thrust::counting_iterator<OffsetType>(0),
                        thrust::counting_iterator<OffsetType>(indices.size()),
                        indices.begin());
}

template <typename IndexArray, typename OffsetArray>
void indices_to_offsets(const IndexArray& indices, OffsetArray& offsets)
{
    typedef typename OffsetArray::value_type OffsetType;

    // convert uncompressed row indices into compressed row offsets
    thrust::lower_bound(indices.begin(),
                        indices.end(),
                        thrust::counting_iterator<OffsetType>(0),
                        thrust::counting_iterator<OffsetType>(offsets.size()),
                        offsets.begin());
}

template<typename T, typename IndexType>
struct row_operator : public std::unary_function<T,IndexType>
{
    row_operator(int a_step)
        : step(a_step)
    {}

    __host__ __device__
        IndexType operator()(const T &value) const
        {
            return value % step;
        }

    private:
    int step; // TODO replace with IndexType
};



// TODO fuse transform and scatter_if together
template <typename Matrix, typename Array>
void extract_diagonal(const Matrix& A, Array& output, cusp::detail::coo_format_tag)
{
    typedef typename Matrix::index_type  IndexType;
    typedef typename Array::value_type   ValueType;
    typedef typename Array::memory_space MemorySpace;  // TODO remove
    
    // initialize output to zero
    thrust::fill(output.begin(), output.end(), ValueType(0));

    // determine which matrix entries correspond to the matrix diagonal
    cusp::array1d<unsigned int,MemorySpace> is_diagonal(A.num_entries);
    thrust::transform(A.row_indices.begin(), A.row_indices.end(), A.column_indices.begin(), is_diagonal.begin(), thrust::equal_to<IndexType>());

    // scatter the diagonal values to output
    thrust::scatter_if(A.values.begin(), A.values.end(),
                       A.row_indices.begin(),
                       is_diagonal.begin(),
                       output.begin());
}


template <typename Matrix, typename Array>
void extract_diagonal(const Matrix& A, Array& output, cusp::detail::csr_format_tag)
{
    typedef typename Matrix::index_type  IndexType;
    typedef typename Array::value_type   ValueType;
    typedef typename Array::memory_space MemorySpace;  // TODO remove
    
    // initialize output to zero
    thrust::fill(output.begin(), output.end(), ValueType(0));

    // first expand the compressed row offsets into row indices
    cusp::array1d<IndexType,MemorySpace> row_indices(A.num_entries);
    offsets_to_indices(A.row_offsets, row_indices);

    // determine which matrix entries correspond to the matrix diagonal
    cusp::array1d<unsigned int,MemorySpace> is_diagonal(A.num_entries);
    thrust::transform(row_indices.begin(), row_indices.end(), A.column_indices.begin(), is_diagonal.begin(), thrust::equal_to<IndexType>());

    // scatter the diagonal values to output
    thrust::scatter_if(A.values.begin(), A.values.end(),
                       row_indices.begin(),
                       is_diagonal.begin(),
                       output.begin());
}


template <typename Matrix, typename Array>
void extract_diagonal(const Matrix& A, Array& output, cusp::detail::dia_format_tag)
{
    typedef typename Matrix::index_type  IndexType;
    typedef typename Array::value_type   ValueType;
    typedef typename Array::memory_space MemorySpace;  // TODO remove
    
    // copy diagonal_offsets to host (sometimes unnecessary)
    cusp::array1d<IndexType,cusp::host_memory> diagonal_offsets(A.diagonal_offsets);

    for(size_t i = 0; i < diagonal_offsets.size(); i++)
    {
        if(diagonal_offsets[i] == 0)
        {
            // diagonal found, copy to output and return
            thrust::copy(A.values.values.begin() + A.values.num_rows * i,
                         A.values.values.begin() + A.values.num_rows * i + output.size(),
                         output.begin());
            return;
        }
    }

    // no diagonal found
    thrust::fill(output.begin(), output.end(), ValueType(0));
}


template <typename Matrix, typename Array>
void extract_diagonal(const Matrix& A, Array& output, cusp::detail::ell_format_tag)
{
    typedef typename Matrix::index_type  IndexType;
    typedef typename Array::value_type   ValueType;
    typedef typename Array::memory_space MemorySpace;  // TODO remove
    
    // initialize output to zero
    thrust::fill(output.begin(), output.end(), ValueType(0));

    // compute ELL row indices
    cusp::array1d<IndexType,MemorySpace> row_indices(A.column_indices.values.size());
    thrust::transform(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(A.column_indices.values.size()),
            row_indices.begin(), row_operator<int,IndexType>(A.column_indices.num_rows));

    // determine which matrix entries correspond to the matrix diagonal
    cusp::array1d<unsigned int,MemorySpace> is_diagonal(A.column_indices.values.size());
    thrust::transform(row_indices.begin(), row_indices.end(), A.column_indices.values.begin(), is_diagonal.begin(), thrust::equal_to<IndexType>());

    // scatter the diagonal values to output
    thrust::scatter_if(A.values.values.begin(), A.values.values.end(),
                       row_indices.begin(),
                       is_diagonal.begin(),
                       output.begin());
}

template <typename Matrix, typename Array>
void extract_diagonal(const Matrix& A, Array& output, cusp::detail::hyb_format_tag)
{
    typedef typename Matrix::index_type  IndexType;
    typedef typename Array::value_type   ValueType;
    typedef typename Array::memory_space MemorySpace;  // TODO remove
    
    // initialize output to zero
    thrust::fill(output.begin(), output.end(), ValueType(0));

    // extract COO diagonal
    {
        // TODO fuse into single operation

        // determine which matrix entries correspond to the matrix diagonal
        cusp::array1d<unsigned int,MemorySpace> is_diagonal(A.coo.num_entries);
        thrust::transform(A.coo.row_indices.begin(), A.coo.row_indices.end(),
                          A.coo.column_indices.begin(),
                          is_diagonal.begin(),
                          thrust::equal_to<IndexType>());
    
        // scatter the diagonal values to output
        thrust::scatter_if(A.coo.values.begin(), A.coo.values.end(),
                           A.coo.row_indices.begin(),
                           is_diagonal.begin(),
                           output.begin());
    }

    // extract ELL diagonal
    {
        // TODO fuse into single operation

        // compute ELL row indices
        cusp::array1d<IndexType,MemorySpace> row_indices(A.ell.column_indices.values.size());
        thrust::transform(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(A.ell.column_indices.values.size()),
                          row_indices.begin(),
                          row_operator<int,IndexType>(A.ell.column_indices.num_rows));

        // determine which matrix entries correspond to the matrix diagonal
        cusp::array1d<unsigned int,MemorySpace> is_diagonal(A.ell.column_indices.values.size());
        thrust::transform(row_indices.begin(), row_indices.end(),
                          A.ell.column_indices.values.begin(),
                          is_diagonal.begin(),
                          thrust::equal_to<IndexType>());

        // scatter the diagonal values to output
        thrust::scatter_if(A.ell.values.values.begin(), A.ell.values.values.end(),
                           row_indices.begin(),
                           is_diagonal.begin(),
                           output.begin());
    }
}


template <typename Matrix, typename Array>
void extract_diagonal(const Matrix& A, Array& output)
{
    output.resize(thrust::min(A.num_rows, A.num_cols));

    // dispatch on matrix_format
    extract_diagonal(A, output, typename cusp::detail::matrix_format<Matrix>::type());
}

} // end namespace detail
} // end namespace cusp

