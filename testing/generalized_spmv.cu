#include <unittest/unittest.h>

#include <cusp/detail/device/generalized_spmv/coo_flat.h>
#include <cusp/detail/device/generalized_spmv/csr_scalar.h>

#include <cusp/array1d.h>
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/multiply.h>
#include <cusp/gallery/poisson.h>
#include <cusp/gallery/random.h>

template <typename Matrix,
          typename Array1,
          typename Array2,
          typename Array3,
          typename UnaryFunction,
          typename BinaryFunction1,
          typename BinaryFunction2>
void generalized_spmv(const Matrix& A,
                      const Array1& x,
                      const Array2& y,
                            Array3& z,
                      UnaryFunction   initialize,
                      BinaryFunction1 combine,
                      BinaryFunction2 reduce,
                      cusp::csr_format)
{
    cusp::detail::device::cuda::spmv_csr_scalar
        (A.num_rows,
         A.row_offsets.begin(), A.column_indices.begin(), A.values.begin(),
         x.begin(), y.begin(), z.begin(),
         initialize, combine, reduce);
}

template <typename Matrix,
          typename Array1,
          typename Array2,
          typename Array3,
          typename UnaryFunction,
          typename BinaryFunction1,
          typename BinaryFunction2>
void generalized_spmv(const Matrix& A,
                      const Array1& x,
                      const Array2& y,
                            Array3& z,
                      UnaryFunction   initialize,
                      BinaryFunction1 combine,
                      BinaryFunction2 reduce,
                      cusp::coo_format)
{
    cusp::detail::device::cuda::spmv_coo
        (A.num_rows, A.num_entries,
         A.row_indices.begin(), A.column_indices.begin(), A.values.begin(),
         x.begin(), y.begin(), z.begin(),
         initialize, combine, reduce);
}

template <typename Matrix,
          typename Array1,
          typename Array2,
          typename Array3,
          typename UnaryFunction,
          typename BinaryFunction1,
          typename BinaryFunction2>
void generalized_spmv(const Matrix& A,
                      const Array1& x,
                      const Array2& y,
                            Array3& z,
                      UnaryFunction   initialize,
                      BinaryFunction1 combine,
                      BinaryFunction2 reduce)
{
  generalized_spmv(A, x, y, z, initialize, combine, reduce, typename Matrix::format());
}

template <typename TestMatrix>
void _TestGeneralizedSpMV(void)
{
  typedef typename TestMatrix::memory_space MemorySpace;
  typedef typename TestMatrix::value_type   ValueType;

  {
    // initialize example matrix
    cusp::array2d<ValueType, cusp::host_memory> A(5,4);
    A(0,0) = 13; A(0,1) = 80; A(0,2) =  0; A(0,3) =  0; 
    A(1,0) =  0; A(1,1) = 27; A(1,2) =  0; A(1,3) =  0;
    A(2,0) = 55; A(2,1) =  0; A(2,2) = 24; A(2,3) = 42;
    A(3,0) =  0; A(3,1) = 69; A(3,2) =  0; A(3,3) = 83;
    A(4,0) =  0; A(4,1) =  0; A(4,2) = 27; A(4,3) =  0;

    // convert to desired format
    TestMatrix test_matrix = A;

    // allocate vectors
    cusp::array1d<ValueType, MemorySpace> x(4);
    cusp::array1d<ValueType, MemorySpace> y(5);
    cusp::array1d<ValueType, MemorySpace> z(5,-1);

    // initialize input and output vectors
    x[0] = 1.0f; y[0] = 10.0f; 
    x[1] = 2.0f; y[1] = 20.0f;
    x[2] = 3.0f; y[2] = 30.0f;
    x[3] = 4.0f; y[3] = 40.0f;
                 y[4] = 50.0f;

    generalized_spmv(test_matrix, x, y, z, thrust::identity<ValueType>(), thrust::multiplies<ValueType>(), thrust::plus<ValueType>());

    ASSERT_EQUAL(z[0], 183.0f);
    ASSERT_EQUAL(z[1],  74.0f);
    ASSERT_EQUAL(z[2], 325.0f);
    ASSERT_EQUAL(z[3], 510.0f);
    ASSERT_EQUAL(z[4], 131.0f);
  }
 
  cusp::array1d<TestMatrix, cusp::host_memory> matrices;

  TestMatrix A; cusp::gallery::poisson5pt(A,   5,   5);
  TestMatrix B; cusp::gallery::poisson5pt(B,  10,  10);
  TestMatrix C; cusp::gallery::poisson5pt(C, 117, 113);
  TestMatrix D; cusp::gallery::random( 21,  23,   5, D);
  TestMatrix E; cusp::gallery::random( 45,  37,  15, E);
  TestMatrix F; cusp::gallery::random(129, 127,  40, F);
  TestMatrix G; cusp::gallery::random(355, 378, 234, G);
    
  matrices.push_back(A);
  matrices.push_back(B);
  matrices.push_back(C);
  matrices.push_back(D);
  matrices.push_back(E);
  matrices.push_back(F);
  matrices.push_back(G);
 
  for(size_t i = 0; i < matrices.size(); i++)
  {
    const TestMatrix& M = matrices[i];

    // allocate vectors
    cusp::array1d<ValueType, MemorySpace> x = unittest::random_samples<bool>(M.num_cols);
    cusp::array1d<ValueType, MemorySpace> y(M.num_rows,0);
    cusp::array1d<ValueType, MemorySpace> z = unittest::random_samples<char>(M.num_rows);

    generalized_spmv(M, x, y, z, thrust::identity<ValueType>(), thrust::multiplies<ValueType>(), thrust::plus<ValueType>());
  
    // compute reference
    cusp::array1d<ValueType, MemorySpace> reference(M.num_rows,0);
    cusp::multiply(M, x, reference);

    ASSERT_EQUAL(z, reference);
  }
}


void TestCsrGeneralizedSpMV(void)
{
  typedef cusp::csr_matrix<int,float,cusp::device_memory> TestMatrix;
  _TestGeneralizedSpMV<TestMatrix>();
}
DECLARE_UNITTEST(TestCsrGeneralizedSpMV);

void TestCooGeneralizedSpMV(void)
{
  typedef cusp::coo_matrix<int,float,cusp::device_memory> TestMatrix;
  _TestGeneralizedSpMV<TestMatrix>();
}
DECLARE_UNITTEST(TestCooGeneralizedSpMV);


