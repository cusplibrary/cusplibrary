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

/*! \file convert.h
 *  \brief Matrix format conversion
 */

#pragma once

#include <cusp/detail/config.h>

namespace cusp
{

/*! \addtogroup algorithms Algorithms
 *  \addtogroup matrix_algorithms Matrix Algorithms
 *  \ingroup algorithms
 *  \brief Algorithms for processing matrices in sparse and dense
 *  formats
 *  \{
 */

/**
 * \brief Convert between matrix formats
 *
 * \tparam SourceType Type of the input matrix to convert
 * \tparam DestinationType Type of the output matrix to create
 *
 * \param src Input matrix to convert
 * \param dst Output matrix created by converting src to the specified format
 *
 * \note DestinationType will be resized as necessary
 *
 * \see \p copy
 */
template <typename SourceType, typename DestinationType>
void convert(const SourceType& src, DestinationType& dst);

/*! \}
 */

} // end namespace cusp

#include <cusp/detail/convert.inl>
