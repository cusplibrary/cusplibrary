#include <unittest/unittest.h>

#include <cusp/csr_matrix.h>
#include <cusp/convert.h>
#include <cusp/memory.h>
#include <cusp/linear_operator.h>


template <class HostMatrix, class TestMatrix>
void _TestMakeLinearOperator(HostMatrix host_matrix, TestMatrix test_matrix)
{
    typedef typename TestMatrix::memory_space MemorySpace;

    cusp::csr_matrix<int, float, cusp::host_memory> csr;
    
    cusp::allocate_matrix(csr, 2, 2, 4);
    
    csr.row_offsets[0] = 0;
    csr.row_offsets[1] = 2;
    csr.row_offsets[2] = 4;
    
    csr.column_indices[0] = 0;   csr.values[0] = 10.0f; 
    csr.column_indices[1] = 1;   csr.values[1] = 11.0f;
    csr.column_indices[2] = 0;   csr.values[2] = 12.0f;
    csr.column_indices[3] = 1;   csr.values[3] = 13.0f;
    
    cusp::convert_matrix(host_matrix, csr); 
    cusp::allocate_matrix_like(test_matrix, host_matrix);
    cusp::memcpy_matrix(test_matrix, host_matrix);

    // allocate vectors
    float * x = cusp::new_array<float, MemorySpace>(2);
    float * y = cusp::new_array<float, MemorySpace>(2);
    
    cusp::set_array_element<MemorySpace>(x, 0, 1.0f);
    cusp::set_array_element<MemorySpace>(x, 1, 2.0f);
    
    cusp::set_array_element<MemorySpace>(y, 0, 100.0f);
    cusp::set_array_element<MemorySpace>(y, 1, 100.0f);

    cusp::make_linear_operator(test_matrix)(x, y);

    ASSERT_EQUAL(cusp::get_array_element<MemorySpace>(y, 0), 132.0f);
    ASSERT_EQUAL(cusp::get_array_element<MemorySpace>(y, 1), 138.0f);

    cusp::deallocate_matrix(csr);
    cusp::deallocate_matrix(host_matrix);
    cusp::deallocate_matrix(test_matrix);
    cusp::delete_array<float, MemorySpace>(x);
    cusp::delete_array<float, MemorySpace>(y);
}

template <class MemorySpace>
void TestCooLinearOperator(void)
{   _TestMakeLinearOperator(cusp::coo_matrix<int, float, cusp::host_memory>(), cusp::coo_matrix<int, float, MemorySpace>());  }
DECLARE_HOST_DEVICE_UNITTEST(TestCooLinearOperator);

template <class MemorySpace>
void TestCsrLinearOperator(void)
{   _TestMakeLinearOperator(cusp::csr_matrix<int, float, cusp::host_memory>(), cusp::csr_matrix<int, float, MemorySpace>());  }
DECLARE_HOST_DEVICE_UNITTEST(TestCsrLinearOperator);

template <class MemorySpace>
void TestDiaLinearOperator(void)
{   _TestMakeLinearOperator(cusp::dia_matrix<int, float, cusp::host_memory>(), cusp::dia_matrix<int, float, MemorySpace>());  }
DECLARE_HOST_DEVICE_UNITTEST(TestDiaLinearOperator);

template <class MemorySpace>
void TestEllLinearOperator(void)
{   _TestMakeLinearOperator(cusp::ell_matrix<int, float, cusp::host_memory>(), cusp::ell_matrix<int, float, MemorySpace>());  }
DECLARE_HOST_DEVICE_UNITTEST(TestEllLinearOperator);

template <class MemorySpace>
void TestHybLinearOperator(void)
{   _TestMakeLinearOperator(cusp::hyb_matrix<int, float, cusp::host_memory>(), cusp::hyb_matrix<int, float, MemorySpace>());  }
DECLARE_HOST_DEVICE_UNITTEST(TestHybLinearOperator);

