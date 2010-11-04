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


#include <cusp/format.h>
#include <cusp/complex.h>
#include <cusp/coo_matrix.h>

#include <iostream>
#include <iomanip>

namespace cusp
{
namespace detail
{

template <typename Matrix>
void print_matrix(const Matrix& A, cusp::coo_format)
{
  std::cout << "sparse matrix <" << A.num_rows << ", " << A.num_cols << "> with " << A.num_entries << " entries\n";

  for(size_t n = 0; n < A.num_entries; n++)
  {
    std::cout << " " << std::setw(14) << A.row_indices[n];
    std::cout << " " << std::setw(14) << A.column_indices[n];
    std::cout << " " << std::setw(14) << A.values[n] << "\n";
  }
}

template <typename Matrix>
void print_matrix(const Matrix& A, cusp::sparse_format)
{
  // general sparse fallback
  cusp::coo_matrix<typename Matrix::index_type, typename Matrix::value_type, cusp::host_memory> coo(A);
  cusp::print_matrix(coo);
}

template <typename Matrix>
void print_matrix(const Matrix& A, cusp::array2d_format)
{
  std::cout << "array2d <" << A.num_rows << ", " << A.num_cols << ">\n";

  for(size_t i = 0; i < A.num_rows; i++)
  {
    for(size_t j = 0; j < A.num_cols; j++)
    {
      std::cout << std::setw(14) << A(i,j);
    }

    std::cout << "\n";
  }
}

template <typename Matrix>
void print_matrix(const Matrix& A, cusp::array1d_format)
{
  std::cout << "array1d <" << A.size() << ">\n";

  for(size_t i = 0; i < A.size(); i++)
    std::cout << std::setw(14) << A[i] << "\n";
}

} // end namespace detail


/////////////////
// Entry Point //
/////////////////

template <typename Matrix>
void print_matrix(const Matrix& A)
{
  cusp::detail::print_matrix(A, typename Matrix::format());
}

} // end namespace cusp

