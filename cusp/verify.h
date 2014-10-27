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

/*! \file verify.h
 *  \brief Validate matrix format
 */

#pragma once

#include <cusp/detail/config.h>

namespace cusp
{

template <typename MatrixType>
bool is_valid_matrix(const MatrixType& A);

template <typename MatrixType, typename OutputStream>
bool is_valid_matrix(const MatrixType& A, OutputStream& ostream);

template <typename MatrixType>
void assert_is_valid_matrix(const MatrixType& A);

template <typename Array1, typename Array2>
void assert_same_dimensions(const Array1& array1,
                            const Array2& array2);

template <typename Array1, typename Array2, typename Array3>
void assert_same_dimensions(const Array1& array1,
                            const Array2& array2,
                            const Array3& array3);

template <typename Array1, typename Array2, typename Array3, typename Array4>
void assert_same_dimensions(const Array1& array1,
                            const Array2& array2,
                            const Array3& array3,
                            const Array4& array4);

} // end namespace cusp

#include <cusp/detail/verify.inl>
