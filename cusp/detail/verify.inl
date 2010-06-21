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

#include <cusp/csr_matrix.h>
#include <cusp/exception.h>

#include <thrust/is_sorted.h>
#include <thrust/count.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>

#include <sstream>

namespace cusp
{
namespace detail
{

///////////////////////////////////
// Helper functions and functors //
///////////////////////////////////

template <typename IndexVector>
thrust::pair<typename IndexVector::value_type, typename IndexVector::value_type>
index_range(const IndexVector& indices)
{
    // return a pair<> containing the min and max value in a range
    thrust::pair<typename IndexVector::const_iterator, typename IndexVector::const_iterator> iter = thrust::minmax_element(indices.begin(), indices.end());
    return thrust::make_pair(*iter.first, *iter.second);
}

template <typename IndexType>
struct is_ell_entry
{
    IndexType num_rows;
    IndexType stride;
    IndexType invalid_index;

    is_ell_entry(IndexType num_rows, IndexType stride, IndexType invalid_index)
        : num_rows(num_rows), stride(stride), invalid_index(invalid_index) {}

    template <typename Tuple>
    __host__ __device__
    bool operator()(const Tuple& t) const
    {
        IndexType n = thrust::get<0>(t);
        IndexType j = thrust::get<1>(t);
        return (n % stride < num_rows) && (j != invalid_index);
    }
};

template <typename IndexType>
struct is_ell_entry_in_bounds
{
    IndexType num_rows;
    IndexType num_cols;
    IndexType stride;
    IndexType invalid_index;

    is_ell_entry_in_bounds(IndexType num_rows, IndexType num_cols, IndexType stride, IndexType invalid_index)
        : num_rows(num_rows), num_cols(num_cols), stride(stride), invalid_index(invalid_index) {}

    template <typename Tuple>
    __host__ __device__
    bool operator()(const Tuple& t) const
    {
        IndexType n = thrust::get<0>(t);
        IndexType j = thrust::get<1>(t);
        return (n % stride < num_rows) && (j != invalid_index) && (j >= 0) && (j < num_cols);
    }
};


///////////////////////////////
// Matrix-Specific Functions //
///////////////////////////////

template <typename IndexType, typename ValueType, typename MemoryType,
          typename OutputStream>
bool is_valid_matrix(const cusp::coo_matrix<IndexType,ValueType,MemoryType>& A,
                           OutputStream& ostream)
{
    // we could relax some of these conditions if necessary
    if (A.row_indices.size() != A.num_entries)
    {
        ostream << "size of row_indices (" << A.row_indices.size() << ") "
                << "should be equal to num_entries (" << A.num_entries << ")";
        return false;
    }
    
    if (A.column_indices.size() != A.num_entries)
    {
        ostream << "size of column_indices (" << A.column_indices.size() << ") "
                << "should be equal to num_entries (" << A.num_entries << ")";
        return false;
    }
    
    if (A.values.size() != A.num_entries)
    {
        ostream << "size of values (" << A.column_indices.size() << ") "
                << "should be equal to num_entries (" << A.num_entries << ")";
        return false;
    }
   
    if (A.num_entries > 0)
    {
        // check that row_indices is a non-decreasing sequence
        if (!thrust::is_sorted(A.row_indices.begin(), A.row_indices.end()))
        {
            ostream << "row indices should form a non-decreasing sequence";
            return false;
        }

        // check that row indices are within [0, num_rows)
        thrust::pair<IndexType,IndexType> min_max_row = index_range(A.row_indices);
        if (min_max_row.first < 0)
        {
            ostream << "row indices should be non-negative";
            return false;
        }
        if (min_max_row.second >= A.num_cols)
        {
            ostream << "row indices should be less than num_row (" << A.num_rows << ")";
            return false;
        }

        // check that column indices are within [0, num_cols)
        thrust::pair<IndexType,IndexType> min_max_col = index_range(A.column_indices);
        if (min_max_col.first < 0)
        {
            ostream << "column indices should be non-negative";
            return false;
        }
        if (min_max_col.second >= A.num_cols)
        {
            ostream << "column indices should be less than num_cols (" << A.num_cols << ")";
            return false;
        }
    }

    return true;
}


template <typename IndexType, typename ValueType, typename MemoryType,
          typename OutputStream>
bool is_valid_matrix(const cusp::csr_matrix<IndexType,ValueType,MemoryType>& A,
                           OutputStream& ostream)
{
    // we could relax some of these conditions if necessary
    
    if (A.row_offsets.size() != A.num_rows + 1)
    {
        ostream << "size of row_offsets (" << A.row_offsets.size() << ") "
                << "should be equal to num_rows + 1 (" << (A.num_rows + 1) << ")";
        return false;
    }
    
    if (A.row_offsets.front() != 0)
    {
        ostream << "first value in row_offsets (" << A.row_offsets.front() << ") "
                << "should be equal to 0";
        return false;
    }

    if (A.row_offsets.back() != A.num_entries)
    {
        ostream << "last value in row_offsets (" << A.row_offsets.back() << ") "
                << "should be equal to num_entries (" << A.num_entries << ")";
        return false;
    }
    
    if (A.column_indices.size() != A.num_entries)
    {
        ostream << "size of column_indices (" << A.column_indices.size() << ") "
                << "should be equal to num_entries (" << A.num_entries << ")";
        return false;
    }
    
    if (A.values.size() != A.num_entries)
    {
        ostream << "size of values (" << A.column_indices.size() << ") "
                << "should be equal to num_entries (" << A.num_entries << ")";
        return false;
    }

    // check that row_offsets is a non-decreasing sequence
    if (!thrust::is_sorted(A.row_offsets.begin(), A.row_offsets.end()))
    {
        ostream << "row offsets should form a non-decreasing sequence";
        return false;
    }

    if (A.num_entries > 0)
    {
        // check that column indices are within [0, num_cols)
        thrust::pair<IndexType,IndexType> min_max = index_range(A.column_indices);

        if (min_max.first < 0)
        {
            ostream << "column indices should be non-negative";
            return false;
        }
        if (min_max.second >= A.num_cols)
        {
            ostream << "column indices should be less than num_cols (" << A.num_cols << ")";
            return false;
        }
    }

    return true;
}


template <typename IndexType, typename ValueType, typename MemoryType,
          typename OutputStream>
bool is_valid_matrix(const cusp::dia_matrix<IndexType,ValueType,MemoryType>& A,
                           OutputStream& ostream)
{
    if (A.num_rows > A.values.num_rows)
    {
        ostream << "number of rows in values array (" << A.values.num_rows << ") ";
        ostream << "should be >= num_rows (" << A.num_rows << ")";
        return false;
    }
    
    if (A.num_rows > A.values.num_rows)
    {
        ostream << "number of rows in values array (" << A.values.num_rows << ") ";
        ostream << "should be >= num_rows (" << A.num_rows << ")";
        return false;
    }

    return true;
}

template <typename IndexType, typename ValueType, typename MemoryType,
          typename OutputStream>
bool is_valid_matrix(const cusp::ell_matrix<IndexType,ValueType,MemoryType>& A,
                           OutputStream& ostream)
{
    const IndexType invalid_index = cusp::ell_matrix<IndexType,ValueType,MemoryType>::invalid_index;

    if (A.column_indices.num_rows != A.values.num_rows ||
        A.column_indices.num_cols != A.values.num_cols)
    {
        ostream << "shape of column_indices array (" << A.column_indices.num_rows << "," << A.column_indices.num_cols << ") ";
        ostream << "should agree with the values array (" << A.values.num_rows << "," << A.values.num_cols << ")";
        return false;
    }
    
    if (A.num_rows > A.values.num_rows)
    {
        ostream << "number of rows in values array (" << A.values.num_rows << ") ";
        ostream << "should be >= num_rows (" << A.num_rows << ")";
        return false;
    }

    // count true number of entries in ell structure
    IndexType true_num_entries = thrust::count_if(
                                    thrust::make_zip_iterator
                                    (
                                        thrust::make_tuple(thrust::counting_iterator<IndexType>(0), 
                                                           A.column_indices.values.begin())
                                    ),
                                    thrust::make_zip_iterator
                                    (
                                        thrust::make_tuple(thrust::counting_iterator<IndexType>(0), 
                                                           A.column_indices.values.begin())
                                    ) + A.column_indices.values.size(),
                                    is_ell_entry<IndexType>(A.num_rows, A.column_indices.num_rows, invalid_index));

    if (A.num_entries != true_num_entries)
    {
        ostream << "number of valid column indices (" << true_num_entries << ") ";
        ostream << "should be == num_entries (" << A.num_entries << ")";
        return false;
    }

    if (A.num_entries > 0)
    {
        // check that column indices are in [0, num_cols)
        IndexType num_entries_in_bounds = thrust::count_if(
                                              thrust::make_zip_iterator
                                              (
                                                  thrust::make_tuple(thrust::counting_iterator<IndexType>(0), 
                                                                     A.column_indices.values.begin())
                                              ),
                                              thrust::make_zip_iterator
                                              (
                                                  thrust::make_tuple(thrust::counting_iterator<IndexType>(0), 
                                                                     A.column_indices.values.begin())
                                              ) + A.column_indices.values.size(),
                                              is_ell_entry_in_bounds<IndexType>(A.num_rows, A.num_cols, A.column_indices.num_rows, invalid_index));
        if (num_entries_in_bounds != true_num_entries)
        {
            ostream << "matrix contains (" << (true_num_entries - num_entries_in_bounds) << ") out-of-bounds column indices";
            return false;
        }
    }

    return true;
}

template <typename IndexType, typename ValueType, typename MemoryType,
          typename OutputStream>
bool is_valid_matrix(const cusp::hyb_matrix<IndexType,ValueType,MemoryType>& A,
                           OutputStream& ostream)
{
    // make sure redundant shapes values agree
    if (A.num_rows != A.ell.num_rows || A.num_rows != A.coo.num_rows ||
        A.num_cols != A.ell.num_cols || A.num_cols != A.coo.num_cols)
    {
        ostream << "matrix shape (" << A.num_rows << "," << A.num_cols << ") ";
        ostream << "should be equal to shape of ELL part (" << A.ell.num_rows << "," << A.ell.num_cols << ") and ";
        ostream << "COO part (" << A.coo.num_rows << "," << A.coo.num_cols << ")";
        return false;
    }

    // check that num_entries = A.ell.num_entries + A.coo.num_entries
    if (A.num_entries != A.ell.num_entries + A.coo.num_entries)
    {
        ostream << "num_entries (" << A.num_entries << ") ";
        ostream << "should be equal to sum of ELL num_entries (" << A.ell.num_entries << ") and ";
        ostream << "COO num_entries (" << A.coo.num_entries << ")";
        return false;
    }

    return cusp::is_valid_matrix(A.ell, ostream) && cusp::is_valid_matrix(A.coo, ostream);
}


template <typename IndexType, typename ValueType, typename MemoryType,
          typename OutputStream>
bool is_valid_matrix(const cusp::array2d<IndexType,ValueType,MemoryType>& A,
                           OutputStream& ostream)
{
    if (A.num_rows * A.num_cols != A.num_entries)
    {
        ostream << "product of matrix dimensions (" << A.num_rows << "," << A.num_cols << ") ";
        ostream << "should equal num_entries (" << A.num_entries << ")";
        return false;
    }
    
    if (A.num_entries != A.values.size())
    {
        ostream << "num_entries (" << A.num_entries << ") ";
        ostream << "should agree with size of values array (" << A.values.size() << ")";
        return false;
    }

    return true;
}

} // end namespace detail


//////////////////
// Entry points //
//////////////////

template <typename MatrixType>
bool is_valid_matrix(const MatrixType& A)
{
    std::ostringstream oss;
    return cusp::is_valid_matrix(A, oss);
}

template <typename MatrixType, typename OutputStream>
bool is_valid_matrix(const MatrixType& A, OutputStream& ostream)
{
    if ((A.num_rows < 0) || (A.num_cols < 0))
    {
        ostream << "matrix has invalid shape (" << A.num_rows << "," << A.num_cols << ")";
        return false;
    }
    
    if (A.num_entries < 0)
    {
        ostream << "matrix has invalid number of entries (" << A.num_entries << ")";
        return false;
    }

    return detail::is_valid_matrix(A, ostream);
}

template <typename MatrixType>
void assert_is_valid_matrix(const MatrixType& A)
{
    std::ostringstream oss;
    bool is_valid = cusp::is_valid_matrix(A, oss);

    if (!is_valid)
        throw cusp::format_exception(oss.str());
}

} // end namespace cusp

