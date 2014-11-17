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
#include <cusp/format.h>
#include <cusp/complex.h>

namespace cusp
{

// Forward definitions
struct row_major;
struct column_major;
template <typename, typename, typename> class array2d;
template <typename, typename, typename> class dia_matrix;
template <typename, typename, typename> class coo_matrix;
template <typename, typename, typename> class csr_matrix;
template <typename, typename, typename> class ell_matrix;
template <typename, typename, typename> class hyb_matrix;

namespace detail
{

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

template<typename MatrixType, typename FormatTag>
struct as_matrix_type
{
    typedef typename get_index_type<MatrixType>::type IndexType;
    typedef typename MatrixType::value_type ValueType;
    typedef typename MatrixType::memory_space MemorySpace;

    typedef typename matrix_type<IndexType,ValueType,MemorySpace,FormatTag>::type type;
};

template<typename MatrixType> struct as_array2d_type : as_matrix_type<MatrixType,array2d_format> {};
template<typename MatrixType> struct as_dia_type : as_matrix_type<MatrixType,dia_format> {};
template<typename MatrixType> struct as_coo_type : as_matrix_type<MatrixType,coo_format> {};
template<typename MatrixType> struct as_csr_type : as_matrix_type<MatrixType,csr_format> {};
template<typename MatrixType> struct as_ell_type : as_matrix_type<MatrixType,ell_format> {};
template<typename MatrixType> struct as_hyb_type : as_matrix_type<MatrixType,hyb_format> {};

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

