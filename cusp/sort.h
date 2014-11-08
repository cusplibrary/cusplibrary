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

/*! \file sort.h
 *  \brief Specialized sorting routines
 */

#pragma once

namespace cusp
{

template <typename ArrayType1, typename ArrayType2, typename ArrayType3>
void sort_by_row(ArrayType1& rows, ArrayType2& columns, ArrayType3& values);

template <typename ArrayType1, typename ArrayType2, typename ArrayType3>
void sort_by_row_and_column(ArrayType1& rows, ArrayType2& columns, ArrayType3& values);

} // end namespace cusp

#include <cusp/detail/sort.inl>

