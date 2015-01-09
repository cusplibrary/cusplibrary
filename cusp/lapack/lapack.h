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

/*! \file lapack.h
 *  \brief Interface to lapack functions
 */

#pragma once

#include <cusp/detail/config.h>

#include <cusp/lapack/detail/defs.h>

namespace cusp
{
namespace lapack
{

template<typename Array2d>
void potrf( Array2d& A );

template<typename Array2d>
void trtri( Array2d& A );

template<typename Array2d, typename Array1d>
void syev( const Array2d& A, Array1d& eigvals, Array2d& eigvecs );

template<typename Array1d1, typename Array1d2, typename Array1d3, typename Array2d>
void stev( const Array1d1& alphas, const Array1d2& betas,
           Array1d3& eigvals, Array2d& eigvecs, char job = EvalsOrEvecs<evecs>::type );

template<typename Array1d1, typename Array1d2, typename Array1d3>
void stev( const Array1d1& alphas, const Array1d2& betas, Array1d3& eigvals );

template<typename Array2d1, typename Array2d2, typename Array1d, typename Array2d3>
void sygv( const Array2d1& A, const Array2d2& B, Array1d& eigvals, Array2d3& eigvecs );

template<typename Array2d, typename Array1d>
void gesv( const Array2d& A, Array2d& B, Array1d& pivots );

} // end namespace lapack
} // end namespace cusp

#include <cusp/lapack/detail/lapack.inl>
