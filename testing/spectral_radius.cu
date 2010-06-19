#include <unittest/unittest.h>

#include <cusp/gallery/poisson.h>
#include <cusp/csr_matrix.h>
#include <cusp/array2d.h>

// take these
#include <cusp/multiply.h>
#include <cusp/array1d.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>

// move these to hash.h or random.h
#include <thrust/detail/integer_traits.h>

// remove this
#include <cusp/print.h>


// http://burtleburtle.net/bob/hash/integer.html
inline
__host__ __device__
unsigned int hash32(unsigned int a)
{
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a <<  5);
    a = (a + 0xd3a2646c) ^ (a <<  9);
    a = (a + 0xfd7046c5) + (a <<  3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

template <typename I, typename T>
struct hash_01
{
    __host__ __device__
    T operator()(const I& index) const
    {
        return T(hash32(index)) / T(thrust::detail::integer_traits<unsigned int>::const_max);
    }
};


template <typename Matrix>    
double estimate_spectral_radius(const Matrix& A, size_t k = 20)
{
    typedef typename Matrix::index_type   IndexType;
    typedef typename Matrix::value_type   ValueType;
    typedef typename Matrix::memory_space MemorySpace;

    const IndexType N = A.num_rows;

    cusp::array1d<ValueType, MemorySpace> x(N);
    cusp::array1d<ValueType, MemorySpace> y(N);

    // initialize x to random values in [0,1)
    thrust::transform(thrust::counting_iterator<IndexType>(0),
                      thrust::counting_iterator<IndexType>(N),
                      x.begin(),
                      hash_01<IndexType,ValueType>());

    for(size_t i = 0; i < k; i++)
    {
        cusp::blas::scal(x, ValueType(1.0) / cusp::blas::nrmmax(x));
        cusp::multiply(A, x, y);
        x.swap(y);
    }
   
    if (k == 0)
        return 0;
    else
        return cusp::blas::nrm2(x) / cusp::blas::nrm2(y);
}

template <class MemorySpace>
void TestEstimateSpectralRadius(void)
{
    // 2x2 diagonal matrix
    {
        cusp::array2d<float, cusp::host_memory> A(2,2);
        A(0,0) = -5; A(0,1) =  0;
        A(1,0) =  0; A(1,1) =  2;
        float rho = 5.0;
        ASSERT_EQUAL((std::abs(estimate_spectral_radius(A) - rho) / rho) < 0.1f, true);
    }

    // 2x2 Poisson problem
    {
        cusp::csr_matrix<int, float, MemorySpace> A; cusp::gallery::poisson5pt(A, 2, 2); 
        float rho = 6.0;
        ASSERT_EQUAL((std::abs(estimate_spectral_radius(A) - rho) / rho) < 0.1f, true);
    }

    // 4x4 Poisson problem
    {
        cusp::csr_matrix<int, float, MemorySpace> A; cusp::gallery::poisson5pt(A, 4, 4); 
        float rho = 7.2360679774997871;
        ASSERT_EQUAL((std::abs(estimate_spectral_radius(A) - rho) / rho) < 0.1f, true);
    }

}
DECLARE_HOST_DEVICE_UNITTEST(TestEstimateSpectralRadius);

