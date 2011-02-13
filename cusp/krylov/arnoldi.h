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

// TODO remove when CUDA 3.2 is unsupported
#if defined(__CUDACC__) && CUDA_VERSION >= 4000
#include <cusp/detail/random.h>
#endif

namespace cusp
{
namespace krylov
{
namespace detail
{

// TODO remove this when CUDA 3.2 is unsupported
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
    srand(0);

    thrust::host_vector<T> vec(N);
    random_sample<T> rnd;

    for(size_t i = 0; i < N; i++)
        vec[i] = rnd();

    return vec;
}

} // end namespace detail

template <typename Matrix, typename Array2d>
void lanczos(const Matrix& A, Array2d& H, size_t k = 10)
{
	typedef typename Matrix::value_type   ValueType;
	typedef typename Matrix::memory_space MemorySpace;

	size_t N = A.num_cols;
	size_t maxiter = std::min(N, k);

    // initialize starting vector to random values in [0,1)
#if defined(__CUDACC__) && CUDA_VERSION >= 4000
    cusp::detail::random_reals<ValueType> random(N);
    cusp::array1d<ValueType,MemorySpace> v1(random);
#else
    // TODO remove when CUDA 3.2 is unsupported
	cusp::array1d<ValueType,MemorySpace> v1 = detail::random_samples<ValueType>(N);
#endif
	cusp::array1d<ValueType,MemorySpace> v0(N);
	cusp::array1d<ValueType,MemorySpace> w(N,0); //TODO remove initialization

	cusp::blas::scal(v1, ValueType(1) / cusp::blas::nrm2(v1));

	Array2d H_(maxiter + 1, maxiter, 0);

	ValueType alpha = 0.0, beta = 0.0;

	size_t j;

	for(j = 0; j < maxiter; j++)
	{
		cusp::multiply(A, v1, w);

		// TODO only enter if statement when A is symmetric. Need to test for symmetry.
		if(j >= 1)
		{
			H_(j - 1, j) = beta;
			cusp::blas::axpy(v0, w, -beta);
		}

		alpha = cusp::blas::dot(w, v1);
		H_(j,j) = alpha;

		cusp::blas::axpy(v1, w, -alpha);

		beta = cusp::blas::nrm2(w);
		H_(j + 1, j) = beta;

		if(beta < 1e-10) break;

		cusp::blas::scal(w, ValueType(1) / beta);				

        // [v0 v1  w] - > [v1  w v0]
        v0.swap(v1);
        v1.swap(w);
	}

	H.resize(j,j);
	for(size_t row = 0; row < j; row++)
		for(size_t col = 0; col < j; col++)
			H(row,col) = H_(row,col);
}

template <typename Matrix, typename Array2d>
void arnoldi(const Matrix& A, Array2d& H, size_t k = 10)
{
	typedef typename Matrix::value_type   ValueType;
	typedef typename Matrix::memory_space MemorySpace;

	size_t N = A.num_rows;

	size_t maxiter = std::min(N, k);

	Array2d H_(maxiter + 1, maxiter, 0);

	std::vector< cusp::array1d<ValueType,MemorySpace> > V(maxiter + 1);
    for (size_t i = 0; i < maxiter + 1; i++)
        V[i].resize(N);
	
    // initialize starting vector to random values in [0,1)
#if defined(__CUDACC__) && CUDA_VERSION >= 4000
    cusp::copy(cusp::detail::random_reals<ValueType>(N), V[0]);
#else
    // TODO remove when CUDA 3.2 is unsupported
    V[0] = detail::random_samples<ValueType>(N);
#endif

    // normalize v0
	cusp::blas::scal(V[0], ValueType(1) / cusp::blas::nrm2(V[0]));	

	size_t j;

	for(j = 0; j < maxiter; j++)
	{
		cusp::multiply(A, V[j], V[j + 1]);

		for(size_t i = 0; i <= j; i++)
		{
			H_(i,j) = cusp::blas::dot(V[i], V[j + 1]);

			cusp::blas::axpy(V[i], V[j + 1], -H_(i,j));
		}

		H_(j+1,j) = cusp::blas::nrm2(V[j + 1]);
		if(H_(j+1,j) < 1e-10) break;

		cusp::blas::scal(V[j + 1], ValueType(1) / H_(j+1,j));
	}

	H.resize(j,j);
	for( size_t row = 0; row < j; row++ )
		for( size_t col = 0; col < j; col++ )
			H(row,col) = H_(row,col);
}

} // end namespace krylov
} // end namespace cusp

