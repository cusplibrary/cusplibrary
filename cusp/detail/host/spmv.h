#pragma once

#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/dia_matrix.h>
#include <cusp/ell_matrix.h>
#include <cusp/hyb_matrix.h>

#include <thrust/functional.h>
#include <cusp/detail/functional.h>

namespace cusp
{
namespace detail
{
namespace host
{

// coo_matrix
template <typename IndexType,
          typename ValueType,
          typename UnaryFunction,
          typename BinaryFunction1,
          typename BinaryFunction2>
void spmv_coo(const IndexType num_rows,
              const IndexType num_cols,
              const IndexType num_entries,
	          const IndexType * row_indices, 
	          const IndexType * column_indices, 
	          const ValueType * values,
	          const ValueType * x,
                    ValueType * y,
              UnaryFunction   initialize,
              BinaryFunction1 combine,
              BinaryFunction2 reduce)
{
    for(IndexType i = 0; i < num_rows; i++)
        y[i] = initialize(y[i]);

    for(IndexType n = 0; n < num_entries; n++)
    {
        const IndexType i   = row_indices[n];
        const IndexType j   = column_indices[n];
        const ValueType Aij = values[n];
        const ValueType xj  = x[j];

        y[i] = reduce(y[i], combine(Aij, xj));
    }
}


template <typename IndexType, typename ValueType>
void spmv(const cusp::coo_matrix<IndexType, ValueType, cusp::host_memory>& coo, 
          const ValueType * x,  
                ValueType * y)
{
    spmv_coo(coo.num_rows, coo.num_cols, coo.num_entries,
             &coo.row_indices[0], &coo.column_indices[0], &coo.values[0],
             x, y,
             cusp::detail::zero_function<ValueType>(), thrust::multiplies<ValueType>(), thrust::plus<ValueType>());
}


// csr_matrix
template <typename IndexType,
          typename ValueType,
          typename UnaryFunction,
          typename BinaryFunction1,
          typename BinaryFunction2>
void spmv_csr(const IndexType num_rows,
              const IndexType num_cols,
              const IndexType * row_offsets,
              const IndexType * column_indices,
              const ValueType * values,
              const ValueType * x,
                    ValueType * y,
              UnaryFunction   initialize,
              BinaryFunction1 combine,
              BinaryFunction2 reduce)
{
 
    for(IndexType i = 0; i < num_rows; i++)
    {
        const IndexType row_start = row_offsets[i];
        const IndexType row_end   = row_offsets[i+1];
 
        ValueType accumulator = initialize(y[i]);
 
        for (IndexType jj = row_start; jj < row_end; jj++)
        {
            const IndexType j   = column_indices[jj];
            const ValueType Aij = values[jj];
            const ValueType xj  = x[j];
 
            accumulator = reduce(accumulator, combine(Aij, xj));
        }
 
        y[i] = accumulator;
    }
}


template <typename IndexType, typename ValueType>
void spmv(const cusp::csr_matrix<IndexType, ValueType, cusp::host_memory>& csr, 
          const ValueType * x,  
                ValueType * y)
{
    spmv_csr(csr.num_rows, csr.num_cols,
             &csr.row_offsets[0], &csr.column_indices[0], &csr.values[0],
             x, y,
             cusp::detail::zero_function<ValueType>(), thrust::multiplies<ValueType>(), thrust::plus<ValueType>());
}


// dia_matrix
template <typename IndexType,
          typename ValueType,
          typename UnaryFunction,
          typename BinaryFunction1,
          typename BinaryFunction2>
void spmv_dia(const IndexType num_rows,
              const IndexType num_cols,
              const IndexType num_diagonals,
              const IndexType stride,
              const IndexType * diagonal_offsets, 
              const ValueType  * values, 
              const ValueType  * x,
                    ValueType  * y,
              UnaryFunction   initialize,
              BinaryFunction1 combine,
              BinaryFunction2 reduce)
{
    for(IndexType i = 0; i < num_rows; i++)
        y[i] = initialize(y[i]);

    for(IndexType i = 0; i < num_diagonals; i++)
    {
        const IndexType k = diagonal_offsets[i];

        const IndexType i_start = std::max<IndexType>(0, -k);
        const IndexType j_start = std::max<IndexType>(0,  k);

        // number of elements to process in this diagonal
        const IndexType N = std::min(num_rows - i_start, num_cols - j_start);

        const ValueType * d_ = values + i*stride + i_start;
        const ValueType * x_ = x + j_start;
              ValueType * y_ = y + i_start;

        for(IndexType n = 0; n < N; n++)
            y_[n] = reduce(y_[n], combine(d_[n], x_[n]));
    }
}

template <typename IndexType, typename ValueType>
void spmv(const dia_matrix<IndexType, ValueType, cusp::host_memory>& dia, 
          const ValueType * x,  
                ValueType * y)
{
    const IndexType num_diagonals = dia.values.num_cols;
    const IndexType stride        = dia.values.num_rows;

    spmv_dia(dia.num_rows, dia.num_cols, num_diagonals, stride,
             &dia.diagonal_offsets[0], &dia.values.values[0],
             x, y,
             cusp::detail::zero_function<ValueType>(), thrust::multiplies<ValueType>(), thrust::plus<ValueType>());
}


// ell_matrix
template <typename IndexType,
          typename ValueType,
          typename UnaryFunction,
          typename BinaryFunction1,
          typename BinaryFunction2>
void spmv_ell(const IndexType num_rows,
              const IndexType num_cols,
              const IndexType num_entries_per_row,
              const IndexType stride,
              const IndexType * column_indices, 
              const ValueType * values, 
              const ValueType * x,
                    ValueType * y,
              UnaryFunction   initialize,
              BinaryFunction1 combine,
              BinaryFunction2 reduce)
{
    const IndexType invalid_index = cusp::ell_matrix<IndexType, ValueType, cusp::host_memory>::invalid_index;
    
    for(IndexType i = 0; i < num_rows; i++)
        y[i] = initialize(y[i]);

    for(IndexType n = 0; n < num_entries_per_row; n++)
    {
        const IndexType * Aj_n = column_indices + n * stride;
        const ValueType * Ax_n = values         + n * stride;

        for(IndexType i = 0; i < num_rows; i++)
        {
            const IndexType j   = Aj_n[i];
            const ValueType Aij = Ax_n[i];

            if (j != invalid_index)
            {
                const ValueType xj  = x[j];
                y[i] = reduce(y[i], combine(Aij, xj));
            }
        }
    }
}

template <typename IndexType, typename ValueType>
void spmv(const cusp::ell_matrix<IndexType, ValueType, cusp::host_memory>& ell, 
          const ValueType * x,  
                ValueType * y)
{
    const IndexType stride              = ell.column_indices.num_rows;
    const IndexType num_entries_per_row = ell.column_indices.num_cols;

    spmv_ell(ell.num_rows, ell.num_cols, num_entries_per_row, stride,
             &ell.column_indices.values[0], &ell.values.values[0],
             x, y,
             cusp::detail::zero_function<ValueType>(), thrust::multiplies<ValueType>(), thrust::plus<ValueType>());
}


// hyb_matrix
template <typename IndexType, typename ValueType>
void spmv(const cusp::hyb_matrix<IndexType, ValueType, cusp::host_memory>& hyb, 
          const ValueType * x,  
                ValueType * y)
{
    const IndexType stride              = hyb.ell.column_indices.num_rows;
    const IndexType num_entries_per_row = hyb.ell.column_indices.num_cols;

    spmv_ell(hyb.ell.num_rows, hyb.ell.num_cols, num_entries_per_row, stride,
             &hyb.ell.column_indices.values[0], &hyb.ell.values.values[0],
             x, y,
             cusp::detail::zero_function<ValueType>(), thrust::multiplies<ValueType>(), thrust::plus<ValueType>());
    spmv_coo(hyb.coo.num_rows, hyb.coo.num_cols, hyb.coo.num_entries,
             &hyb.coo.row_indices[0], &hyb.coo.column_indices[0], &hyb.coo.values[0],
             x, y,
             thrust::identity<ValueType>(), thrust::multiplies<ValueType>(), thrust::plus<ValueType>());
}

} // end namespace host
} // end namespace detail
} // end namespace cusp

