#include <unittest/unittest.h>

#include <cusp/array2d.h>

#include <cusp/print.h> // TODO remove

#include <cmath>

template <typename IndexType, typename ValueType, typename SpaceOrAlloc, typename Orientation>
int lu_factor(cusp::array2d<ValueType,SpaceOrAlloc,Orientation>& A,
              cusp::array1d<IndexType,SpaceOrAlloc>& pivot)
{
    const int n = A.num_rows;

    // For each row and column, k = 0, ..., n-1,
    for (int k = 0; k < n; k++)
    {
        ValueType * p_k = &A.values[0] + k * n;

        // find the pivot row
        pivot[k] = k;
        ValueType max = std::fabs(A(k,k));
        
        for (int j = k + 1; j < n; j++)
        {
            if (max < std::fabs(A(j,k)))
            {
                max = std::fabs(A(j,k));
                pivot[k] = j;
            }
        }

        // and if the pivot row differs from the current row, then
        // interchange the two rows.
        if (pivot[k] != k)
            for (int j = 0; j < n; j++)
                std::swap(A(k,j), A(pivot[k],j));

        // and if the matrix is singular, return error
        if (A(k,k) == 0.0)
            return -1;

        // otherwise find the lower triangular matrix elements for column k. 
        for (int i = k + 1; i < n; i++)
            A(i,k) /= A(k,k);

        // update remaining matrix
        for (int i = k + 1; i < n; i++)
            for (int j = k + 1; j < n; j++)
                A(i,j) -= A(i,k) * A(k,j);
    }

    return 0;
}



template <typename IndexType, typename ValueType, typename SpaceOrAlloc, typename Orientation>
int lu_solve(const cusp::array2d<ValueType,SpaceOrAlloc,Orientation>& A,
             const cusp::array1d<ValueType,SpaceOrAlloc>& b,
             const cusp::array1d<IndexType,SpaceOrAlloc>& pivot,
                   cusp::array1d<ValueType,SpaceOrAlloc>& x)
{
    const int n = A.num_rows;
   
    // copy rhs to x
    x = b;

    // Solve the linear equation Lx = b for x, where L is a lower
    // triangular matrix with an implied 1 along the diagonal.
    for (int k = 0; k < n; k++)
    {
        if (pivot[k] != k)
            std::swap(x[k],x[pivot[k]]);

        for (int i = 0; i < k; i++)
            x[k] -= A(k,i) * x[i];
    }

    // Solve the linear equation Ux = y, where y is the solution
    // obtained above of Lx = b and U is an upper triangular matrix.
    for (int k = n - 1; k >= 0; k--)
    {
        for (int i = k + 1; i < n; i++)
            x[k] -= A(k,i) * x[i];

        if (A(k,k) == 0)
            return -1;

        x[k] /= A(k,k);
    }

    return 0;
}


void TestSolveArray2d(void)
{
    cusp::array2d<float, cusp::host_memory> A(4,4);
    A(0,0) = 0.83228434;  A(0,1) = 0.41106598;  A(0,2) = 0.72609841;  A(0,3) = 0.80428486;
    A(1,0) = 0.00890590;  A(1,1) = 0.29940800;  A(1,2) = 0.60630740;  A(1,3) = 0.33654542;
    A(2,0) = 0.22525064;  A(2,1) = 0.93054253;  A(2,2) = 0.37939225;  A(2,3) = 0.16235888;
    A(3,0) = 0.83911960;  A(3,1) = 0.21176293;  A(3,2) = 0.21010691;  A(3,3) = 0.52911885;
    
    cusp::array1d<float, cusp::host_memory> b(4);
    b[0] = 1.31699541; 
    b[1] = 0.87768331;
    b[2] = 1.18994714;
    b[3] = 0.61914723;
   
    std::cout << "\nA" << std::endl;
    cusp::print_matrix(A);
    std::cout << "b" << std::endl;
    cusp::print_matrix(b);
    
    cusp::array1d<int, cusp::host_memory>   pivot(4);
    cusp::array1d<float, cusp::host_memory> x(4);
    lu_factor(A, pivot);
    lu_solve(A, b, pivot, x);

    std::cout << "LU" << std::endl;
    cusp::print_matrix(A);
    std::cout << "pivot" << std::endl;
    cusp::print_matrix(pivot);
    std::cout << "x" << std::endl;
    cusp::print_matrix(x);
    
    cusp::array1d<float, cusp::host_memory> expected(4);
    expected[0] = 0.21713221; 
    expected[1] = 0.80528582;
    expected[2] = 0.98416811;
    expected[3] = 0.11271028;

    ASSERT_EQUAL(std::fabs(expected[0] - x[0]) < 1e-4, true);
    ASSERT_EQUAL(std::fabs(expected[1] - x[1]) < 1e-4, true);
    ASSERT_EQUAL(std::fabs(expected[2] - x[2]) < 1e-4, true);
    ASSERT_EQUAL(std::fabs(expected[3] - x[3]) < 1e-4, true);
}
DECLARE_UNITTEST(TestSolveArray2d);


