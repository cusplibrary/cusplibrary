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

/*! \file permutation_matrix.h
 *  \brief Compressed Sparse Row matrix format.
 */

#pragma once

#include <cusp/detail/config.h>

#include <cusp/array1d.h>
#include <cusp/format.h>
#include <cusp/detail/matrix_base.h>

namespace cusp
{

// forward definition
template <typename Array, typename IndexType, typename MemorySpace> class permutation_matrix_view;

/*! \addtogroup sparse_matrices Sparse Matrices
 */

/*! \addtogroup sparse_matrix_containers Sparse Matrix Containers
 *  \ingroup sparse_matrices
 *  \{
 */

/*! \p permutation_matrix : Permutation matrix container
 *
 * \tparam IndexType Type used for matrix indices (e.g. \c int).
 * \tparam ValueType Type used for matrix values (e.g. \c float).
 * \tparam MemorySpace A memory space (e.g. \c cusp::host_memory or cusp::device_memory)
 *
 * \note The matrix entries within the same row must be sorted by column index.
 * \note The matrix should not contain duplicate entries.
 *
 *  The following code snippet demonstrates how to create a 3-by-3
 *  \p permutation_matrix on the host with 3 nonzeros and then copies the
 *  matrix to the device.
 *
 *  \code
 *  #include <cusp/permutation_matrix.h>
 *  ...
 *
 *  // allocate storage for (4,4) matrix with 4 nonzeros
 *  cusp::permutation_matrix<int,cusp::host_memory> A(4);
 *
 *  // initialize matrix entries on host
 *  A.permutation[0] = 2; // first offset is always zero
 *  A.permutation[1] = 1;
 *  A.permutation[2] = 0; // last offset is always num_entries
 *
 *
 *  // A now represents the following matrix
 *  //    [10  0 20]
 *  //    [ 0  0  0]
 *  //    [ 0  0 30]
 *
 *  // copy to the device
 *  cusp::permutation_matrix<int,cusp::device_memory> B = A;
 *  \endcode
 *
 */
template <typename IndexType, class MemorySpace>
class permutation_matrix : public detail::matrix_base<IndexType,IndexType,MemorySpace,cusp::permutation_format>
{
  typedef cusp::detail::matrix_base<IndexType,IndexType,MemorySpace,cusp::permutation_format> Parent;
  public:
    /*! rebind matrix to a different MemorySpace
     */
    template<typename MemorySpace2>
    struct rebind { typedef cusp::permutation_matrix<IndexType, MemorySpace2> type; };

    /*! type of permutation indices array
     */
    typedef typename cusp::array1d<IndexType, MemorySpace> permutation_array_type;

    /*! equivalent container type
     */
    typedef typename cusp::permutation_matrix<IndexType, MemorySpace> container;

    /*! equivalent view type
     */
    typedef typename cusp::permutation_matrix_view<typename permutation_array_type::view,
                                           IndexType, MemorySpace> view;

    /*! equivalent const_view type
     */
    typedef typename cusp::permutation_matrix_view<typename permutation_array_type::const_view,
                                                    IndexType, MemorySpace> const_view;

    /*! Storage for the permutation indices
     */
    permutation_array_type permutation;

    /*! Construct an empty \p permutation_matrix.
     */
    permutation_matrix() {}

    /*! Construct a \p permutation_matrix with a specific shape and number of nonzero entries.
     *
     *  \param num_rows Number of rows.
     *  \param num_cols Number of columns.
     *  \param num_entries Number of nonzero matrix entries.
     */
    permutation_matrix(size_t num_rows)
      : Parent(num_rows, num_rows, num_rows),
        permutation(cusp::counting_array<int>(num_rows)) {}

    /*! Construct a \p permutation_matrix from another matrix.
     *
     *  \param matrix Another sparse or dense matrix.
     */
    template<typename MemorySpace2>
    permutation_matrix(const permutation_matrix<IndexType,MemorySpace2>& P)
      : Parent(P), permutation(P.permutation) {}

    /*! Construct a \p permutation_matrix from another matrix.
     *
     *  \param matrix Another sparse or dense matrix.
     */
    template<typename Array1>
    permutation_matrix(size_t num_rows, const Array1& permutation)
      : Parent(num_rows, num_rows, num_rows), permutation(permutation) {}

    /*! Resize matrix dimensions and underlying storage
     */
    void resize(size_t num_rows)
    {
      Parent::resize(num_rows, num_rows, num_rows);
      permutation.resize(num_rows);
    }

    /*! Swap the contents of two \p permutation_matrix objects.
     *
     *  \param matrix Another \p permutation_matrix with the same IndexType and ValueType.
     */
    void swap(permutation_matrix& matrix)
    {
      Parent::swap(matrix);
      permutation.swap(matrix.permutation);
    }

    /*! Permute rows and columns of matrix elements
     */
    template<typename MatrixType>
    void symmetric_permute(MatrixType& A);
}; // class permutation_matrix
/*! \}
 */

/*! \addtogroup sparse_matrix_views Sparse Matrix Views
 *  \ingroup sparse_matrices
 *  \{
 */

/*! \p permutation_matrix_view : Compressed Sparse Row matrix view
 *
 * \tparam Array Type of \c permutation array view
 * \tparam IndexType Type used for matrix indices (e.g. \c int).
 * \tparam MemorySpace A memory space (e.g. \c cusp::host_memory or cusp::device_memory)
 *
 */
template <typename Array,
          typename IndexType   = typename Array::value_type,
          typename MemorySpace = typename Array::memory_space>
class permutation_matrix_view : public cusp::detail::matrix_base<IndexType,IndexType,MemorySpace,cusp::permutation_format>
{
  typedef cusp::detail::matrix_base<IndexType,IndexType,MemorySpace,cusp::permutation_format> Parent;
  public:
    typedef Array permutation_array_type;

    /*! equivalent container type
     */
    typedef typename cusp::permutation_matrix<IndexType, MemorySpace> container;

    /*! equivalent view type
     */
    typedef typename cusp::permutation_matrix_view<Array, IndexType, MemorySpace> view;

    /*! Storage for the permutation indices
     */
    permutation_array_type permutation;

    // construct empty view
    permutation_matrix_view(void)
      : Parent() {}

    // construct from existing permutation matrix or view
    permutation_matrix_view(permutation_matrix<IndexType,MemorySpace>& P)
      : Parent(P),
        permutation(P.permutation) {}

    // TODO check sizes here
    template<typename Array1>
    permutation_matrix_view(size_t num_rows, const Array1& permutation)
      : Parent(num_rows, num_rows, num_rows),
        permutation(permutation) {}

    /*! Resize matrix dimensions and underlying storage
     */
    void resize(size_t num_rows)
    {
      Parent::resize(num_rows, num_rows, num_rows);
      permutation.resize(num_rows);
    }

    /*! Permute rows and columns of matrix elements
     */
    template<typename MatrixType>
    void symmetric_permute(MatrixType& A);
};

/* Convenience functions */

template <typename Array>
permutation_matrix_view<Array>
make_permutation_matrix_view(size_t num_rows, Array permutation)
{
  return permutation_matrix_view<Array>
    (num_rows, permutation);
}

template <typename Array,
          typename IndexType,
          typename MemorySpace>
permutation_matrix_view<Array,IndexType,MemorySpace>
make_permutation_matrix_view(const permutation_matrix_view<Array,IndexType,MemorySpace>& m)
{
  return permutation_matrix_view<Array,IndexType,MemorySpace>(m);
}

template <typename IndexType, class MemorySpace>
typename permutation_matrix<IndexType,MemorySpace>::view
make_permutation_matrix_view(permutation_matrix<IndexType,MemorySpace>& m)
{
  return make_permutation_matrix_view
    (m.num_rows, make_array1d_view(m.permutation));
}

template <typename IndexType, class MemorySpace>
typename permutation_matrix<IndexType,MemorySpace>::const_view
make_permutation_matrix_view(const permutation_matrix<IndexType,MemorySpace>& m)
{
  return make_permutation_matrix_view
    (m.num_rows, make_array1d_view(m.permutation));
}
/*! \}
 */

} // end namespace cusp

#include <cusp/detail/permutation_matrix.inl>
