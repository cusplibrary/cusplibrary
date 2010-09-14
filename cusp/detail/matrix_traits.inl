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

#include <cusp/detail/forward_definitions.h>

namespace cusp
{
namespace detail
{

struct known_format_tag   {};
struct unknown_format_tag {};

struct dense_format_tag : public known_format_tag {};
struct array1d_format_tag : public dense_format_tag {};
struct array2d_format_tag : public dense_format_tag {};

struct sparse_format_tag : public known_format_tag {};
struct coo_format_tag : public sparse_format_tag {};
struct csr_format_tag : public sparse_format_tag {};
struct dia_format_tag : public sparse_format_tag {};
struct ell_format_tag : public sparse_format_tag {};
struct hyb_format_tag : public sparse_format_tag {};

template <typename T> struct matrix_format { typedef unknown_format_tag type; };
template <typename T> struct matrix_format<const T> : public matrix_format<T> { };
template <typename ValueType, typename MemorySpace>                       struct matrix_format< cusp::array1d<ValueType,MemorySpace> >              { typedef array1d_format_tag type; };
template <typename ValueType, typename MemorySpace, typename Orientation> struct matrix_format< cusp::array2d<ValueType,MemorySpace,Orientation> >  { typedef array2d_format_tag type; };
template <typename IndexType, typename ValueType, typename MemorySpace>   struct matrix_format< cusp::coo_matrix<IndexType,ValueType,MemorySpace> > { typedef coo_format_tag     type; };
template <typename IndexType, typename ValueType, typename MemorySpace>   struct matrix_format< cusp::csr_matrix<IndexType,ValueType,MemorySpace> > { typedef csr_format_tag     type; };
template <typename IndexType, typename ValueType, typename MemorySpace>   struct matrix_format< cusp::dia_matrix<IndexType,ValueType,MemorySpace> > { typedef dia_format_tag     type; };
template <typename IndexType, typename ValueType, typename MemorySpace>   struct matrix_format< cusp::ell_matrix<IndexType,ValueType,MemorySpace> > { typedef ell_format_tag     type; };
template <typename IndexType, typename ValueType, typename MemorySpace>   struct matrix_format< cusp::hyb_matrix<IndexType,ValueType,MemorySpace> > { typedef hyb_format_tag     type; };

} // end namespace detail
} // end namespace cusp

