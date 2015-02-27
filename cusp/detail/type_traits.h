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


/*! \file type_traits.h
 *  \brief Temporarily define some type traits
 *         until nvcc can compile tr1::type_traits.
 */

#pragma once

#include <cusp/detail/config.h>
#include <cusp/detail/format.h>
#include <cusp/detail/functional.h>
#include <cusp/complex.h>

#include <cusp/iterator/join_iterator.h>

namespace cusp
{

// Forward definitions
struct row_major;
struct column_major;
template <typename, typename>           class array1d;
template <typename, typename, typename> class array2d;
template <typename, typename, typename> class dia_matrix;
template <typename, typename, typename> class coo_matrix;
template <typename, typename, typename> class csr_matrix;
template <typename, typename, typename> class ell_matrix;
template <typename, typename, typename> class hyb_matrix;

template <typename ArrayType1, typename ArrayType2, typename ArrayType3,
          typename IndexType, typename ValueType, typename MemorySpace> class coo_matrix_view;

namespace detail
{

template <typename IndexType,typename Orientation1,typename Orientation2>struct logical_to_other_physical_functor;

template<typename MatrixType, typename CompareTag>
struct is_matrix_type : thrust::detail::integral_constant<bool,
        thrust::detail::is_same<typename MatrixType::format,CompareTag>::value>
{};

template<typename MatrixType> struct is_array2d : is_matrix_type<MatrixType,array2d_format> {};
template<typename MatrixType> struct is_coo     : is_matrix_type<MatrixType,coo_format> {};
template<typename MatrixType> struct is_csr     : is_matrix_type<MatrixType,csr_format> {};
template<typename MatrixType> struct is_dia     : is_matrix_type<MatrixType,dia_format> {};
template<typename MatrixType> struct is_ell     : is_matrix_type<MatrixType,ell_format> {};
template<typename MatrixType> struct is_hyb     : is_matrix_type<MatrixType,hyb_format> {};

template<typename IndexType, typename ValueType, typename MemorySpace, typename FormatTag> struct matrix_type {};

template<typename IndexType, typename ValueType, typename MemorySpace>
struct matrix_type<IndexType,ValueType,MemorySpace,array1d_format>
{
    typedef cusp::array1d<ValueType,MemorySpace> type;
};

template<typename IndexType, typename ValueType, typename MemorySpace>
struct matrix_type<IndexType,ValueType,MemorySpace,array2d_format>
{
    typedef cusp::array2d<ValueType,MemorySpace,cusp::row_major> type;
};

template<typename IndexType, typename ValueType, typename MemorySpace>
struct matrix_type<IndexType,ValueType,MemorySpace,dia_format>
{
    typedef cusp::dia_matrix<IndexType,ValueType,MemorySpace> type;
};

template<typename IndexType, typename ValueType, typename MemorySpace>
struct matrix_type<IndexType,ValueType,MemorySpace,coo_format>
{
    typedef cusp::coo_matrix<IndexType,ValueType,MemorySpace> type;
};

template<typename IndexType, typename ValueType, typename MemorySpace>
struct matrix_type<IndexType,ValueType,MemorySpace,csr_format>
{
    typedef cusp::csr_matrix<IndexType,ValueType,MemorySpace> type;
};

template<typename IndexType, typename ValueType, typename MemorySpace>
struct matrix_type<IndexType,ValueType,MemorySpace,ell_format>
{
    typedef cusp::ell_matrix<IndexType,ValueType,MemorySpace> type;
};

template<typename IndexType, typename ValueType, typename MemorySpace>
struct matrix_type<IndexType,ValueType,MemorySpace,hyb_format>
{
    typedef cusp::hyb_matrix<IndexType,ValueType,MemorySpace> type;
};

template<typename MatrixType, typename Format = typename MatrixType::format>
struct get_index_type
{
    typedef typename MatrixType::index_type type;
};

template<typename MatrixType>
struct get_index_type<MatrixType,array1d_format>
{
    typedef int type;
};

template<typename MatrixType, typename MemorySpace, typename FormatTag>
struct as_matrix_type
{
    typedef typename get_index_type<MatrixType>::type IndexType;
    typedef typename MatrixType::value_type ValueType;

    typedef typename matrix_type<IndexType,ValueType,MemorySpace,FormatTag>::type type;
};

template<typename MatrixType,typename MemorySpace=typename MatrixType::memory_space>
struct as_array2d_type : as_matrix_type<MatrixType,MemorySpace,array2d_format> {};

template<typename MatrixType,typename MemorySpace=typename MatrixType::memory_space>
struct as_dia_type : as_matrix_type<MatrixType,MemorySpace,dia_format> {};

template<typename MatrixType,typename MemorySpace=typename MatrixType::memory_space>
struct as_coo_type : as_matrix_type<MatrixType,MemorySpace,coo_format> {};

template<typename MatrixType,typename MemorySpace=typename MatrixType::memory_space>
struct as_csr_type : as_matrix_type<MatrixType,MemorySpace,csr_format> {};

template<typename MatrixType,typename MemorySpace=typename MatrixType::memory_space>
struct as_ell_type : as_matrix_type<MatrixType,MemorySpace,ell_format> {};

template<typename MatrixType,typename MemorySpace=typename MatrixType::memory_space>
struct as_hyb_type : as_matrix_type<MatrixType,MemorySpace,hyb_format> {};

template<typename IndexType,typename ValueType,typename MemorySpace,typename FormatTag>
struct coo_view_type{};

template<typename IndexType,typename ValueType,typename MemorySpace>
struct coo_view_type<IndexType,ValueType,MemorySpace,hyb_format>
{
    typedef cusp::ell_matrix<IndexType,ValueType,MemorySpace>                                               EllMatrixType;
    typedef thrust::counting_iterator<IndexType>                                                            CountingIterator;
    typedef thrust::transform_iterator<cusp::detail::divide_value<IndexType>, CountingIterator>             RowIndexIterator;
    typedef typename EllMatrixType::column_indices_array_type::values_array_type::iterator                  ColumnIndexIterator;
    typedef typename EllMatrixType::values_array_type::values_array_type::iterator                          ValueIterator;

    typedef cusp::detail::logical_to_other_physical_functor<IndexType, cusp::row_major, cusp::column_major> PermFunctor;
    typedef thrust::transform_iterator<PermFunctor, CountingIterator>                                       PermIndexIterator;
    typedef thrust::permutation_iterator<ColumnIndexIterator, PermIndexIterator>                            PermColumnIndicesIterator;
    typedef thrust::permutation_iterator<ValueIterator, PermIndexIterator>                                  PermValueIterator;

    typedef typename cusp::array1d<IndexType,MemorySpace>::iterator                                         IndexIterator;
    typedef cusp::join_iterator<RowIndexIterator,ColumnIndexIterator,IndexIterator>                         JoinRowIterator;
    typedef cusp::join_iterator<PermColumnIndicesIterator,ColumnIndexIterator,IndexIterator>                JoinColumnIterator;
    typedef cusp::join_iterator<PermValueIterator,ValueIterator,IndexIterator>                              JoinValueIterator;

    typedef cusp::array1d_view<typename JoinRowIterator::iterator>                                          Array1;
    typedef cusp::array1d_view<typename JoinColumnIterator::iterator>                                       Array2;
    typedef cusp::array1d_view<typename JoinValueIterator::iterator>                                        Array3;

    typedef cusp::coo_matrix_view<Array1,Array2,Array3,IndexType,ValueType,MemorySpace>                     type;
};

template<typename T1, typename T2, typename T = void>
struct enable_if_same_system
        : thrust::detail::enable_if< thrust::detail::is_same<typename T1::memory_space,typename T2::memory_space>::value, T >
{};
template<typename T1, typename T2, typename T = void>
struct enable_if_different_system
        : thrust::detail::enable_if< thrust::detail::is_different<typename T1::memory_space,typename T2::memory_space>::value, T >
{};

template <typename T>
struct norm_type
{
    typedef T type;
};

template <typename T>
struct norm_type< cusp::complex<T> >
{
    typedef T type;
};

} // end detail
} // end cusp

