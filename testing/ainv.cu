#include <unittest/unittest.h>

#include <cusp/precond/ainv.h>

#include <cusp/gallery/poisson.h>
#include <cusp/krylov/cg.h>

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

struct hash_01
{
    __host__ __device__
    float operator()(const unsigned int& index) const
    {
        return (float)(hash32(index)) / ((float)0xffffffff);
    }
};

void TestAINV(void)
{
    typedef int                 IndexType;
    typedef float               ValueType;
    typedef cusp::device_memory MemorySpace;

    // Create 2D Poisson problem
    cusp::csr_matrix<IndexType,ValueType,MemorySpace> A;
    cusp::gallery::poisson5pt(A, 100, 100);
    A.values[0] = 10;
    int N = A.num_rows;

    cusp::array1d<ValueType,MemorySpace> x(N);
    cusp::array1d<ValueType,MemorySpace> b(N, 0);
    
    thrust::transform(thrust::counting_iterator<unsigned int>(0),
                      thrust::counting_iterator<unsigned int>(N),
                      x.begin(),
                      hash_01());



    // test as preconditioner
    {
        cusp::array1d<ValueType,MemorySpace> x_solve = x;
        cusp::precond::bridson_ainv<ValueType,MemorySpace> M(A, .1);

        cusp::default_monitor<ValueType> monitor(b, 1000, 0, 1e-5);
        cusp::krylov::cg(A, x_solve, b, monitor, M);

        ASSERT_EQUAL(monitor.converged(), true);
    }
}
DECLARE_UNITTEST(TestAINV);

