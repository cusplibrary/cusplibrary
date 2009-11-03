#include <unittest/unittest.h>

#include <cusp/csr_matrix.h>
#include <cusp/array1d.h>
#include <cusp/linear_operator.h>

template <class TestMatrix>
void _TestMakeLinearOperator(TestMatrix test_matrix)
{
    typedef typename TestMatrix::memory_space MemorySpace;

    cusp::csr_matrix<int, float, cusp::host_memory> csr(2, 2, 4);
    
    csr.row_offsets[0] = 0;
    csr.row_offsets[1] = 2;
    csr.row_offsets[2] = 4;
    
    csr.column_indices[0] = 0;   csr.values[0] = 10.0f; 
    csr.column_indices[1] = 1;   csr.values[1] = 11.0f;
    csr.column_indices[2] = 0;   csr.values[2] = 12.0f;
    csr.column_indices[3] = 1;   csr.values[3] = 13.0f;
    
    test_matrix = csr;

    // allocate vectors
    cusp::array1d<float, MemorySpace> x(2);
    cusp::array1d<float, MemorySpace> y(2);
    
    x[0] = 1.0f;
    x[1] = 2.0f;
    
    y[0] = 100.0f;
    y[1] = 100.0f;

    cusp::make_linear_operator(test_matrix)(thrust::raw_pointer_cast(&x[0]),
                                            thrust::raw_pointer_cast(&y[0]));

    ASSERT_EQUAL(y[0], 132.0f);
    ASSERT_EQUAL(y[1], 138.0f);
}

template <class MemorySpace>
void TestCooLinearOperator(void)
{   _TestMakeLinearOperator(cusp::coo_matrix<int, float, MemorySpace>());  }
DECLARE_HOST_DEVICE_UNITTEST(TestCooLinearOperator);

template <class MemorySpace>
void TestCsrLinearOperator(void)
{   _TestMakeLinearOperator(cusp::csr_matrix<int, float, MemorySpace>());  }
DECLARE_HOST_DEVICE_UNITTEST(TestCsrLinearOperator);

template <class MemorySpace>
void TestDiaLinearOperator(void)
{   _TestMakeLinearOperator(cusp::dia_matrix<int, float, MemorySpace>());  }
DECLARE_HOST_DEVICE_UNITTEST(TestDiaLinearOperator);

template <class MemorySpace>
void TestEllLinearOperator(void)
{   _TestMakeLinearOperator(cusp::ell_matrix<int, float, MemorySpace>());  }
DECLARE_HOST_DEVICE_UNITTEST(TestEllLinearOperator);

template <class MemorySpace>
void TestHybLinearOperator(void)
{   _TestMakeLinearOperator(cusp::hyb_matrix<int, float, MemorySpace>());  }
DECLARE_HOST_DEVICE_UNITTEST(TestHybLinearOperator);

