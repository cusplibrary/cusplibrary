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

#include <cusp/multiply.h>
#include <cusp/array1d.h>

#define DEFAULT_SEED 0

namespace cusp
{
namespace krylov
{
namespace detail
{

template<typename ValueType>
  struct random_sample
{
  double operator()(void) const
  {
    return ValueType(20.0) * (rand() / (ValueType(RAND_MAX) + ValueType(1.0))) - ValueType(10.0);
  }
};


template<typename T>
thrust::host_vector<T> random_samples(const size_t N)
{
    srand(DEFAULT_SEED);

    thrust::host_vector<T> vec(N);
    random_sample<T> rnd;

    for(size_t i = 0; i < N; i++)
        vec[i] = rnd();

    return vec;
}

} // end namespace detail

template <typename Matrix, typename Array2d>
void lanczos( const Matrix& A, Array2d& H, size_t k = 10 ){

	typedef typename Matrix::index_type   IndexType;
	typedef typename Matrix::value_type   ValueType;
	typedef typename Matrix::memory_space MemorySpace;

	size_t N = A.num_cols;
	IndexType maxiter = std::min( N, k );
	cusp::array1d<ValueType,MemorySpace> v0 = detail::random_samples<ValueType>(N);

	ValueType norm_v0 = cusp::blas::nrm2(v0);
	cusp::blas::scal( v0, ValueType(1.0)/norm_v0 );	

	Array2d H_(maxiter+1,maxiter,0);
	std::vector< cusp::array1d<ValueType,MemorySpace> > V;
	V.push_back(v0);

	cusp::array1d<ValueType,MemorySpace> w(N,0);
	ValueType alpha = 0.0, beta = 0.0;

	IndexType j;
	for( j = 0; j < maxiter; j++ )
	{
		cusp::multiply(A,V.back(),w);

		// TODO only enter if statement when A is symmetric. Need to test for symmetry.
		if( j >= 1 )
		{
			H_(j-1,j) = beta;
			cusp::blas::axpy(V.front(),w,-beta);
		}

		alpha = cusp::blas::dot(w,V.back());
		H_(j,j) = alpha;	
		
		cusp::blas::axpy(V.back(),w,-alpha);

		beta = cusp::blas::nrm2(w);
		H_(j+1,j) = beta;

		if( H_(j+1,j) < 1e-10 ) break;

		cusp::blas::scal( w, ValueType(1.0)/beta );				

		// swap the front and back vectors
		// then swap in the new vector while avoiding explicit copying
		V.front().swap(V.back());
		V.back().swap(w);
	}

	H.resize(j,j);
	for( IndexType row = 0; row < j; row++ )
		for( IndexType col = 0; col < j; col++ )
			H(row,col) = H_(row,col);
}

template <typename Matrix, typename Array2d>
void arnoldi( const Matrix& A, Array2d& H, size_t k = 10 )
{
	typedef typename Matrix::index_type   IndexType;
	typedef typename Matrix::value_type   ValueType;
	typedef typename Matrix::memory_space MemorySpace;

	size_t N = A.num_rows;

	size_t maxiter = std::min( N, k );
	cusp::array1d<ValueType,MemorySpace> v0 = detail::random_samples<ValueType>(N);

	ValueType norm_v0 = cusp::blas::nrm2(v0);
	cusp::blas::scal( v0, ValueType(1.0)/norm_v0 );	

	Array2d H_(maxiter+1,maxiter,0);
	std::vector< cusp::array1d<ValueType,MemorySpace> > V;
	V.push_back(v0);

	cusp::array1d<ValueType,MemorySpace> w(N,0);

	size_t j;

	for( j = 0; j < maxiter; j++ )
	{
		cusp::multiply(A,V.back(),w);

		for(size_t i = 0; i < V.size(); i++ )
		{
			H_(i,j) =  cusp::blas::dot( V.at(i), w );
			cusp::blas::axpy(V.at(i),w,-H_(i,j));
		}

		H_(j+1,j) = cusp::blas::nrm2(w);
		if( H_(j+1,j) < 1e-10 ) break;

		cusp::blas::scal( w, ValueType(1.0)/H_(j+1,j) );
		V.push_back(w);
	}

	H.resize(j,j);
	for( size_t row = 0; row < j; row++ )
		for( size_t col = 0; col < j; col++ )
			H(row,col) = H_(row,col);
}

} // end namespace krylov
} // end namespace cusp

