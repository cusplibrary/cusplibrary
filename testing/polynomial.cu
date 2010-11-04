#include <unittest/unittest.h>

#include <cusp/relaxation/polynomial.h>
#include <cusp/detail/spectral_radius.h>

#include <cusp/array2d.h>
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/dia_matrix.h>
#include <cusp/ell_matrix.h>
#include <cusp/hyb_matrix.h>

template <typename Matrix>
void TestPolynomialRelaxation(void)
{
    typedef typename Matrix::memory_space Space;

    cusp::array2d<float, Space> M(5,5);
    M(0,0) = 1.0;  M(0,1) = 1.0;  M(0,2) = 2.0;  M(0,3) = 0.0;  M(0,4) = 0.0; 
    M(1,0) = 3.0;  M(1,1) = 2.0;  M(1,2) = 0.0;  M(1,3) = 0.0;  M(1,4) = 5.0;
    M(2,0) = 0.0;  M(2,1) = 0.0;  M(2,2) = 0.5;  M(2,3) = 0.0;  M(2,4) = 0.0;
    M(3,0) = 0.0;  M(3,1) = 6.0;  M(3,2) = 7.0;  M(3,3) = 4.0;  M(3,4) = 0.0;
    M(4,0) = 0.0;  M(4,1) = 8.0;  M(4,2) = 0.0;  M(4,3) = 0.0;  M(4,4) = 8.0;

    cusp::array1d<float, Space> b(5,  5.0);
    cusp::array1d<float, Space> x(5, -1.0);
    cusp::array1d<float, Space> expected(5);
    expected[0] =  3.35407;  
    expected[1] =  0.70829;  
    expected[2] =  4.07994;  
    expected[3] = -2.68237;  
    expected[4] = -1.14735;  


    Matrix A(M);
    cusp::array1d<float,cusp::host_memory> coeff;
    float rho = cusp::detail::estimate_spectral_radius(A);
    cusp::relaxation::detail::chebyshev_polynomial_coefficients(rho,coeff);
    cusp::relaxation::polynomial<float, Space> relax(A, coeff);

    relax(A, b, x);

    ASSERT_ALMOST_EQUAL(x, expected);
}
DECLARE_SPARSE_MATRIX_UNITTEST(TestPolynomialRelaxation);

