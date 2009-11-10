#include <unittest/unittest.h>

#include <cusp/csr_matrix.h>
#include <cusp/array1d.h>
#include <cusp/spblas.h>

template <class TestMatrix>
void _TestSparseSpMV(TestMatrix test_matrix)
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

    cusp::spblas::spmv(test_matrix, x, y);

    ASSERT_EQUAL(y[0], 132.0f);
    ASSERT_EQUAL(y[1], 138.0f);
}

template <class MemorySpace>
void TestCooSpMV(void)
{   _TestSparseSpMV(cusp::coo_matrix<int, float, MemorySpace>());  }
DECLARE_HOST_DEVICE_UNITTEST(TestCooSpMV);

template <class MemorySpace>
void TestCsrSpMV(void)
{   _TestSparseSpMV(cusp::csr_matrix<int, float, MemorySpace>());  }
DECLARE_HOST_DEVICE_UNITTEST(TestCsrSpMV);

template <class MemorySpace>
void TestDiaSpMV(void)
{   _TestSparseSpMV(cusp::dia_matrix<int, float, MemorySpace>());  }
DECLARE_HOST_DEVICE_UNITTEST(TestDiaSpMV);

template <class MemorySpace>
void TestEllSpMV(void)
{   _TestSparseSpMV(cusp::ell_matrix<int, float, MemorySpace>());  }
DECLARE_HOST_DEVICE_UNITTEST(TestEllSpMV);

template <class MemorySpace>
void TestHybSpMV(void)
{   _TestSparseSpMV(cusp::hyb_matrix<int, float, MemorySpace>());  }
DECLARE_HOST_DEVICE_UNITTEST(TestHybSpMV);

