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


#pragma once

#include <cusp/detail/config.h>

namespace cusp
{
namespace io
{

template <typename MatrixType>
void read_matrix_market_file(MatrixType& mtx, const std::string& filename);

template <typename MatrixType>
void write_matrix_market_file(const MatrixType& mtx, const std::string& filename);

} //end namespace io
} //end namespace cusp

#include <cusp/io/detail/matrix_market.inl>

