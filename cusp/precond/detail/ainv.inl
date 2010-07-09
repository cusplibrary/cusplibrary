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

/*! \file ainv.inl
 *  \brief Inline file for ainv.h
 */

#include <cusp/blas.h>
#include <cusp/detail/format_utils.h>
#include <cusp/transpose.h>
#include <cusp/multiply.h>
#include <cusp/csr_matrix.h>

#include <map>
#include <vector>

namespace cusp
{
namespace precond
{

namespace detail
{

template<typename IndexType, typename ValueType>
void vector_scalar(std::map<IndexType, ValueType> &vec, ValueType scalar)
{
    for (typename std::map<IndexType, ValueType>::iterator vec_iter = vec.begin(); vec_iter != vec.end(); ++vec_iter) {
      vec_iter->second *= scalar;
    }
}

template<typename IndexType, typename ValueType>
void matrix_vector_product(const csr_matrix<IndexType, ValueType, host_memory> &A, const std::map<IndexType, ValueType> &x, std::map<IndexType, ValueType> &b)
{
    b.clear();

    for (typename std::map<IndexType, ValueType>::const_iterator x_iter = x.begin(); x_iter != x.end(); ++x_iter) {
        ValueType x_i  = x_iter->second;
        IndexType row = x_iter->first;

        IndexType row_start = A.row_offsets[row];
        IndexType row_end = A.row_offsets[row+1];

        for (IndexType row_j = row_start; row_j < row_end; row_j++) {
            IndexType col = A.column_indices[row_j];
            ValueType Aij = A.values[row_j];

            ValueType product = Aij * x_i;

            // add to b if it's not already in b
            typename std::map<IndexType, ValueType>::iterator b_iter = b.find(col);
            if (b_iter == b.end())
                b[col] = product;
            else 
                b_iter->second += product;
        }
    }

}


template<typename IndexType, typename ValueType>
ValueType dot_product(const std::map<IndexType, ValueType> &a, const std::map<IndexType, ValueType> &b) 
{
    typename std::map<IndexType, ValueType>::const_iterator a_iter = a.begin();
    typename std::map<IndexType, ValueType>::const_iterator b_iter = b.begin();

    ValueType sum = 0;
    while (a_iter != a.end() && b_iter != b.end()) {
        IndexType a_ind = a_iter->first;
        IndexType b_ind = b_iter->first;
        if (a_ind == b_ind) {
            sum += a_iter->second * b_iter->second;
            ++a_iter;
            ++b_iter;
        }
        else if (a_ind < b_ind) 
            ++a_iter;
        else 
            ++b_iter;
    }

    return sum;
}


template<typename IndexType, typename ValueType>
void vector_add_inplace_drop(std::map<IndexType, ValueType> &result, ValueType mult, const std::map<IndexType, ValueType> &operand, ValueType tolerance)
{
    // write into result:
    // result += mult * operand
    // but dropping any terms from (mult * operand) if they are less than tolerance

    for (typename std::map<IndexType, ValueType>::const_iterator op_iter = operand.begin(); op_iter != operand.end(); ++op_iter) {
        IndexType i = op_iter->first;
        ValueType term = mult * op_iter->second;

        // we could replace this with a bound on the size of result.  if we've already reached that maximum size
        // and this would add a new entry to result, we add it only if it is larger than one of the current entries 
        // in result, in which case we remove that element in its place.  This is similar to the Lin algorithm.
        // it would be an interesting experiment to compare different drop tolerances against this method...
        ValueType abs_term = term < 0 ? -term : term;

        if (abs_term < tolerance)
            continue;

        typename std::map<IndexType, ValueType>::iterator result_iter = result.find(i);
        if (result_iter == result.end())
            result[i] = term;
        else
            result_iter->second += term;
    }
}



} // end namespace detail


// constructor
template <typename ValueType, typename MemorySpace>
    template<typename MatrixTypeA>
    bridson_ainv<ValueType,MemorySpace>
    ::bridson_ainv(const MatrixTypeA & A, ValueType drop_tolerance)
        : linear_operator<ValueType,MemorySpace>(A.num_rows, A.num_cols, A.num_rows)
    {
        typename MatrixTypeA::index_type n = A.num_rows;
  
        // copy A to host
        typename csr_matrix<typename MatrixTypeA::index_type, typename MatrixTypeA::value_type, host_memory> host_A = A;
        array1d<ValueType, host_memory> host_diagonals(n);
        

        // perform factorization
        typename std::vector<std::map<typename MatrixTypeA::index_type, typename MatrixTypeA::value_type> > w_factor(n);

        typename MatrixTypeA::index_type i,j;
        for (i=0; i < n; i++) {
          w_factor[i][i] = (typename MatrixTypeA::value_type)1; 
        }

        typename std::map<typename MatrixTypeA::index_type, typename MatrixTypeA::value_type> u;

        for (j=0; j < n; j++) {
          detail::matrix_vector_product(host_A, w_factor[j], u);
          typename MatrixTypeA::value_type p = detail::dot_product(w_factor[j], u);
          host_diagonals[j] = (ValueType) (1.0/p);

          // for i = j+1 to n, skipping where u_i == 0
          // this should be a O(1)-time operation, since u is a sparse vector
          for (typename std::map<typename MatrixTypeA::index_type,typename MatrixTypeA::value_type>::const_iterator u_iter = u.upper_bound(j); u_iter != u.end(); ++u_iter) {
            i = u_iter->first;
            detail::vector_add_inplace_drop(w_factor[i], -u_iter->second/p, w_factor[j], drop_tolerance);
          }

        }

        // copy w_factor into w, w_t
        diagonals = host_diagonals;

        // calculate nnz
        typename MatrixTypeA::index_type nnz = 0;
        for (i=0; i < n; i++)
          nnz += w_factor[i].size();

        csr_matrix<typename MatrixTypeA::index_type, typename MatrixTypeA::value_type, host_memory> host_w(n, n, nnz);

        typename MatrixTypeA::index_type pos = 0;
        host_w.row_offsets[0] = 0;

        for (i=0; i < n; i++) {
          typename std::map<typename MatrixTypeA::index_type, typename MatrixTypeA::value_type>::const_iterator w_iter = w_factor[i].begin();
          while (w_iter != w_factor[i].end()) {
            host_w.column_indices[pos] = w_iter->first;
            host_w.values        [pos] = w_iter->second;

            ++w_iter;
            ++pos;
          }
          host_w.row_offsets[i+1] = pos;
        }

        w = host_w;
        transpose(w, w_t);
    }
        
// linear operator
template <typename ValueType, typename MemorySpace>
    template <typename VectorType1, typename VectorType2>
    void bridson_ainv<ValueType, MemorySpace>
    ::operator()(const VectorType1& x, VectorType2& y) const
    {
        VectorType2 temp1(x.size()), temp2(x.size());
        cusp::multiply(w, x, temp1);
        cusp::blas::xmy(temp1, diagonals, temp2);
        cusp::multiply(w_t, temp2, y);
    }




// constructor
template <typename ValueType, typename MemorySpace>
    template<typename MatrixTypeA>
    scaled_bridson_ainv<ValueType,MemorySpace>
    ::scaled_bridson_ainv(const MatrixTypeA & A, ValueType drop_tolerance)
        : linear_operator<ValueType,MemorySpace>(A.num_rows, A.num_cols, A.num_rows)
    {
        typename MatrixTypeA::index_type n = A.num_rows;
  
        // copy A to host
        typename csr_matrix<typename MatrixTypeA::index_type, typename MatrixTypeA::value_type, host_memory> host_A = A;
        
        // perform factorization
        typename std::vector<std::map<typename MatrixTypeA::index_type, typename MatrixTypeA::value_type> > w_factor(n);

        typename MatrixTypeA::index_type i,j;
        for (i=0; i < n; i++) {
          w_factor[i][i] = (typename MatrixTypeA::value_type)1; 
        }

        typename std::map<typename MatrixTypeA::index_type, typename MatrixTypeA::value_type> u;

        for (j=0; j < n; j++) {
          detail::matrix_vector_product(host_A, w_factor[j], u);
          typename MatrixTypeA::value_type p = detail::dot_product(w_factor[j], u);

          detail::vector_scalar(u, (typename MatrixTypeA::value_type) (1.0/sqrt((ValueType) p)));
          detail::vector_scalar(w_factor[j], (typename MatrixTypeA::value_type) (1.0/sqrt((ValueType) p)));

          // for i = j+1 to n, skipping where u_i == 0
          // this should be a O(1)-time operation, since u is a sparse vector
          for (typename std::map<typename MatrixTypeA::index_type,typename MatrixTypeA::value_type>::const_iterator u_iter = u.upper_bound(j); u_iter != u.end(); ++u_iter) {
            i = u_iter->first;
            detail::vector_add_inplace_drop(w_factor[i], -u_iter->second, w_factor[j], drop_tolerance);
          }

        }

        // copy w_factor into w:

        // calculate nnz
        typename MatrixTypeA::index_type nnz = 0;
        for (i=0; i < n; i++)
          nnz += w_factor[i].size();

        csr_matrix<typename MatrixTypeA::index_type, typename MatrixTypeA::value_type, host_memory> host_w(n, n, nnz);

        typename MatrixTypeA::index_type pos = 0;
        host_w.row_offsets[0] = 0;

        for (i=0; i < n; i++) {
          typename std::map<typename MatrixTypeA::index_type, typename MatrixTypeA::value_type>::const_iterator w_iter = w_factor[i].begin();
          while (w_iter != w_factor[i].end()) {
            host_w.column_indices[pos] = w_iter->first;
            host_w.values        [pos] = w_iter->second;

            ++w_iter;
            ++pos;
          }
          host_w.row_offsets[i+1] = pos;
        }

        w = host_w;

        // w_t is the transpose
        transpose(w, w_t);
    }

template <typename ValueType, typename MemorySpace>
    template <typename VectorType1, typename VectorType2>
    void scaled_bridson_ainv<ValueType, MemorySpace>
    ::operator()(const VectorType1& x, VectorType2& y) const
    {
        VectorType2 temp1(x.size());
        cusp::multiply(w, x, temp1);
        cusp::multiply(w_t, temp1, y);
    }


} // end namespace precond
} // end namespace cusp

