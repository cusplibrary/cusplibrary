#include <unittest/unittest.h>

#include <cusp/precond/aggregation/smoothed_aggregation.h>

#include <cusp/array2d.h>
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/dia_matrix.h>
#include <cusp/ell_matrix.h>
#include <cusp/hyb_matrix.h>

#include <cusp/gallery/poisson.h>
#include <cusp/krylov/cg.h>
#include <cusp/print.h>

template <class MemorySpace>
void TestStandardAggregation(void)
{
    // TODO make this test something, possibly disjoint things that must aggregate

    typedef typename cusp::precond::aggregation::select_sa_matrix_type<int,float,MemorySpace>::type SetupMatrixType;

    SetupMatrixType A;
    cusp::gallery::poisson5pt(A, 10, 10);

    cusp::array1d<int,MemorySpace> aggregates(A.num_rows);
    cusp::precond::aggregation::standard_aggregation(A, aggregates);
}
DECLARE_HOST_DEVICE_UNITTEST(TestStandardAggregation);


template <class MemorySpace>
void TestEstimateRhoDinvA(void)
{
    // 2x2 diagonal matrix
    {
        cusp::csr_matrix<int, float, MemorySpace> A(2,2,2);
        A.row_offsets[0] = 0;
        A.row_offsets[1] = 1;
        A.row_offsets[2] = 2;
        A.column_indices[0] = 0;
        A.column_indices[1] = 1;
        A.values[0] = -5;
        A.values[1] =  2;
        float rho = 1.0;
        ASSERT_EQUAL((std::abs(cusp::precond::aggregation::detail::estimate_rho_Dinv_A(A) - rho) / rho) < 0.1f, true);
    }

    // 2x2 Poisson problem
    {
        cusp::csr_matrix<int, float, MemorySpace> A;
        cusp::gallery::poisson5pt(A, 2, 2);
        float rho = 1.5;
        ASSERT_EQUAL((std::abs(cusp::precond::aggregation::detail::estimate_rho_Dinv_A(A) - rho) / rho) < 0.1f, true);
    }

    // 4x4 Poisson problem
    {
        cusp::csr_matrix<int, float, MemorySpace> A;
        cusp::gallery::poisson5pt(A, 4, 4);
        float rho = 1.8090169943749468;
        ASSERT_EQUAL((std::abs(cusp::precond::aggregation::detail::estimate_rho_Dinv_A(A) - rho) / rho) < 0.1f, true);
    }
}
DECLARE_HOST_DEVICE_UNITTEST(TestEstimateRhoDinvA);


template <typename MemorySpace>
void TestFitCandidates(void)
{
    typedef typename cusp::precond::aggregation::select_sa_matrix_type<int,float,MemorySpace>::type SetupMatrixType;

    // 2 aggregates with 2 nodes each
    {
        cusp::array1d<int,MemorySpace> aggregates(4);
        aggregates[0] = 0;
        aggregates[1] = 0;
        aggregates[2] = 1;
        aggregates[3] = 1;
        cusp::array1d<float,MemorySpace> B(4);
        B[0] = 0.0f;
        B[1] = 1.0f;
        B[2] = 3.0f;
        B[3] = 4.0f;

        SetupMatrixType Q;
        cusp::array1d<float,MemorySpace> R(2);

        cusp::precond::aggregation::detail::fit_candidates(aggregates, B, Q, R);

        ASSERT_EQUAL(R[0], 1.0f);
        ASSERT_EQUAL(R[1], 5.0f);
        ASSERT_ALMOST_EQUAL(Q.values[0], 0.0f);
        ASSERT_ALMOST_EQUAL(Q.values[1], 1.0f);
        ASSERT_ALMOST_EQUAL(Q.values[2], 0.6f);
        ASSERT_ALMOST_EQUAL(Q.values[3], 0.8f);
    }

    // 4 aggregates with varying numbers of nodes
    {
        cusp::array1d<int,MemorySpace> aggregates(10);
        aggregates[0] = 1;
        aggregates[1] = 2;
        aggregates[2] = 0;
        aggregates[3] = 3;
        aggregates[4] = 0;
        aggregates[5] = 2;
        aggregates[6] = 1;
        aggregates[7] = 2;
        aggregates[8] = 1;
        aggregates[9] = 1;
        cusp::array1d<float,MemorySpace> B(10,1.0f);

        SetupMatrixType Q;
        cusp::array1d<float,MemorySpace> R(4);

        cusp::precond::aggregation::detail::fit_candidates(aggregates, B, Q, R);

        ASSERT_ALMOST_EQUAL(R[0], 1.41421f);
        ASSERT_ALMOST_EQUAL(R[1], 2.00000f);
        ASSERT_ALMOST_EQUAL(R[2], 1.73205f);
        ASSERT_ALMOST_EQUAL(R[3], 1.00000f);

        ASSERT_ALMOST_EQUAL(Q.values[0], 0.500000f);
        ASSERT_ALMOST_EQUAL(Q.values[1], 0.577350f);
        ASSERT_ALMOST_EQUAL(Q.values[2], 0.707107f);
        ASSERT_ALMOST_EQUAL(Q.values[3], 1.000000f);
        ASSERT_ALMOST_EQUAL(Q.values[4], 0.707107f);
        ASSERT_ALMOST_EQUAL(Q.values[5], 0.577350f);
        ASSERT_ALMOST_EQUAL(Q.values[6], 0.500000f);
        ASSERT_ALMOST_EQUAL(Q.values[7], 0.577350f);
        ASSERT_ALMOST_EQUAL(Q.values[8], 0.500000f);
        ASSERT_ALMOST_EQUAL(Q.values[9], 0.500000f);
    }

    // TODO test case w/ unaggregated nodes (marked w/ -1)
}
DECLARE_HOST_DEVICE_UNITTEST(TestFitCandidates);


template <class MemorySpace>
void TestSmoothProlongator(void)
{
    typedef typename cusp::precond::aggregation::select_sa_matrix_type<int,float,MemorySpace>::type SetupMatrixType;

    // simple example with diagonal S
    {
        cusp::coo_matrix<int,float,MemorySpace> _S(4,4,4);
        _S.row_indices[0] = 0;
        _S.column_indices[0] = 0;
        _S.values[0] = 1;
        _S.row_indices[1] = 1;
        _S.column_indices[1] = 1;
        _S.values[1] = 2;
        _S.row_indices[2] = 2;
        _S.column_indices[2] = 2;
        _S.values[2] = 3;
        _S.row_indices[3] = 3;
        _S.column_indices[3] = 3;
        _S.values[3] = 4;
        SetupMatrixType S(_S);

        cusp::coo_matrix<int,float,MemorySpace> _T(4,2,4);
        _T.row_indices[0] = 0;
        _T.column_indices[0] = 0;
        _T.values[0] = 0.5;
        _T.row_indices[1] = 1;
        _T.column_indices[1] = 0;
        _T.values[1] = 0.5;
        _T.row_indices[2] = 2;
        _T.column_indices[2] = 1;
        _T.values[2] = 0.5;
        _T.row_indices[3] = 3;
        _T.column_indices[3] = 1;
        _T.values[3] = 0.5;
        SetupMatrixType T(_T);

        SetupMatrixType _P;

        cusp::precond::aggregation::smooth_prolongator(S, T, _P, 4.0f, 2.0f);

        cusp::coo_matrix<int,float,MemorySpace> P(_P);

        ASSERT_EQUAL(P.num_rows,    4);
        ASSERT_EQUAL(P.num_cols,    2);
        ASSERT_EQUAL(P.num_entries, 4);

        ASSERT_EQUAL(P.row_indices[0], 0);
        ASSERT_EQUAL(P.column_indices[0], 0);
        ASSERT_EQUAL(P.row_indices[1], 1);
        ASSERT_EQUAL(P.column_indices[1], 0);
        ASSERT_EQUAL(P.row_indices[2], 2);
        ASSERT_EQUAL(P.column_indices[2], 1);
        ASSERT_EQUAL(P.row_indices[3], 3);
        ASSERT_EQUAL(P.column_indices[3], 1);

        ASSERT_EQUAL(P.values[0], -0.5);
        ASSERT_EQUAL(P.values[1], -0.5);
        ASSERT_EQUAL(P.values[2], -0.5);
        ASSERT_EQUAL(P.values[3], -0.5);
    }

    // 1D Poisson problem w/ 4 points and 2 aggregates
    {
        cusp::coo_matrix<int,float,MemorySpace> _S(4,4,10);
        _S.row_indices[0] = 0;
        _S.column_indices[0] = 0;
        _S.values[0] = 2;
        _S.row_indices[1] = 0;
        _S.column_indices[1] = 1;
        _S.values[1] =-1;
        _S.row_indices[2] = 1;
        _S.column_indices[2] = 0;
        _S.values[2] =-1;
        _S.row_indices[3] = 1;
        _S.column_indices[3] = 1;
        _S.values[3] = 2;
        _S.row_indices[4] = 1;
        _S.column_indices[4] = 2;
        _S.values[4] =-1;
        _S.row_indices[5] = 2;
        _S.column_indices[5] = 1;
        _S.values[5] =-1;
        _S.row_indices[6] = 2;
        _S.column_indices[6] = 2;
        _S.values[6] = 2;
        _S.row_indices[7] = 2;
        _S.column_indices[7] = 3;
        _S.values[7] =-1;
        _S.row_indices[8] = 3;
        _S.column_indices[8] = 2;
        _S.values[8] =-1;
        _S.row_indices[9] = 3;
        _S.column_indices[9] = 3;
        _S.values[9] = 2;
        SetupMatrixType S(_S);

        cusp::coo_matrix<int,float,MemorySpace> _T(4,2,4);
        _T.row_indices[0] = 0;
        _T.column_indices[0] = 0;
        _T.values[0] = 0.5;
        _T.row_indices[1] = 1;
        _T.column_indices[1] = 0;
        _T.values[1] = 0.5;
        _T.row_indices[2] = 2;
        _T.column_indices[2] = 1;
        _T.values[2] = 0.5;
        _T.row_indices[3] = 3;
        _T.column_indices[3] = 1;
        _T.values[3] = 0.5;
        SetupMatrixType T(_T);

        SetupMatrixType _P;

        cusp::precond::aggregation::smooth_prolongator(S, T, _P, 4.0f/3.0f, 1.8090169943749472f);

        cusp::coo_matrix<int,float,MemorySpace> P(_P);
        P.sort_by_row_and_column();

        ASSERT_EQUAL(P.num_rows,    4);
        ASSERT_EQUAL(P.num_cols,    2);
        ASSERT_EQUAL(P.num_entries, 6);

        ASSERT_EQUAL(P.row_indices[0], 0);
        ASSERT_EQUAL(P.column_indices[0], 0);
        ASSERT_EQUAL(P.row_indices[1], 1);
        ASSERT_EQUAL(P.column_indices[1], 0);
        ASSERT_EQUAL(P.row_indices[2], 1);
        ASSERT_EQUAL(P.column_indices[2], 1);
        ASSERT_EQUAL(P.row_indices[3], 2);
        ASSERT_EQUAL(P.column_indices[3], 0);
        ASSERT_EQUAL(P.row_indices[4], 2);
        ASSERT_EQUAL(P.column_indices[4], 1);
        ASSERT_EQUAL(P.row_indices[5], 3);
        ASSERT_EQUAL(P.column_indices[5], 1);

        ASSERT_ALMOST_EQUAL(P.values[0], 0.31573787f);
        ASSERT_ALMOST_EQUAL(P.values[1], 0.31573787f);
        ASSERT_ALMOST_EQUAL(P.values[2], 0.18426213f);
        ASSERT_ALMOST_EQUAL(P.values[3], 0.18426213f);
        ASSERT_ALMOST_EQUAL(P.values[4], 0.31573787f);
        ASSERT_ALMOST_EQUAL(P.values[5], 0.31573787f);
    }
}
DECLARE_HOST_DEVICE_UNITTEST(TestSmoothProlongator);

template <typename SparseMatrix>
void TestSmoothedAggregation(void)
{
    typedef typename SparseMatrix::index_type   IndexType;
    typedef typename SparseMatrix::value_type   ValueType;
    typedef typename SparseMatrix::memory_space MemorySpace;

    // Create 2D Poisson problem
    SparseMatrix A;
    cusp::gallery::poisson5pt(A, 100, 100);

    // create smoothed aggregation solver
    cusp::precond::aggregation::smoothed_aggregation<IndexType,ValueType,MemorySpace> M(A);

    // test as standalone solver
    {
        cusp::array1d<ValueType,MemorySpace> b = unittest::random_samples<ValueType>(A.num_rows);
        cusp::array1d<ValueType,MemorySpace> x = unittest::random_samples<ValueType>(A.num_rows);

        // set stopping criteria (iteration_limit = 40, relative_tolerance = 1e-4)
        cusp::monitor<ValueType> monitor(b, 40, 1e-4);
        M.solve(b,x,monitor);

        ASSERT_EQUAL(monitor.converged(), true);
        ASSERT_EQUAL(monitor.geometric_rate() < 0.8, true);
    }

    // test as preconditioner
    {
        cusp::array1d<ValueType,MemorySpace> b = unittest::random_samples<ValueType>(A.num_rows);
        cusp::array1d<ValueType,MemorySpace> x = unittest::random_samples<ValueType>(A.num_rows);

        // set stopping criteria (iteration_limit = 20, relative_tolerance = 1e-4)
        cusp::monitor<ValueType> monitor(b, 20, 1e-4);
        cusp::krylov::cg(A, x, b, monitor, M);

        ASSERT_EQUAL(monitor.converged(), true);
        ASSERT_EQUAL(monitor.geometric_rate() < 0.5, true);
    }
}
DECLARE_SPARSE_MATRIX_UNITTEST(TestSmoothedAggregation);

void TestSmoothedAggregationHostToDevice(void)
{
    typedef int                 IndexType;
    typedef float               ValueType;

    // Create 2D Poisson problem
    cusp::coo_matrix<IndexType,ValueType,cusp::host_memory> A_h;
    cusp::gallery::poisson5pt(A_h, 100, 100);

    // create smoothed aggregation solver
    cusp::precond::aggregation::smoothed_aggregation<IndexType,ValueType,cusp::host_memory> M_h(A_h);
    cusp::precond::aggregation::smoothed_aggregation<IndexType,ValueType,cusp::device_memory> M_d(M_h);

    // test as standalone solver
    {
        cusp::array1d<ValueType,cusp::device_memory> b = unittest::random_samples<ValueType>(A_h.num_rows);
        cusp::array1d<ValueType,cusp::device_memory> x = unittest::random_samples<ValueType>(A_h.num_rows);

        // set stopping criteria (iteration_limit = 40, relative_tolerance = 1e-4)
        cusp::monitor<ValueType> monitor(b, 40, 1e-4);
        M_d.solve(b,x,monitor);

        ASSERT_EQUAL(monitor.converged(), true);
        ASSERT_EQUAL(monitor.geometric_rate() < 0.8, true);
    }

    // test as preconditioner
    {
        cusp::coo_matrix<IndexType,ValueType,cusp::device_memory> A_d(A_h);
        cusp::array1d<ValueType,cusp::device_memory> b = unittest::random_samples<ValueType>(A_d.num_rows);
        cusp::array1d<ValueType,cusp::device_memory> x = unittest::random_samples<ValueType>(A_d.num_rows);

        // set stopping criteria (iteration_limit = 20, relative_tolerance = 1e-4)
        cusp::monitor<ValueType> monitor(b, 20, 1e-4);
        cusp::krylov::cg(A_d, x, b, monitor, M_d);

        ASSERT_EQUAL(monitor.converged(), true);
        ASSERT_EQUAL(monitor.geometric_rate() < 0.5, true);
    }
}
DECLARE_UNITTEST(TestSmoothedAggregationHostToDevice);


template <typename SparseMatrix>
void TestSymmetricStrengthOfConnection(void)
{
    typedef cusp::array2d<float,cusp::host_memory> Matrix;

    // input
    Matrix M(4,4);
    M(0,0) =  3;
    M(0,1) =  0;
    M(0,2) =  1;
    M(0,3) =  2;
    M(1,0) =  0;
    M(1,1) =  4;
    M(1,2) =  3;
    M(1,3) =  4;
    M(2,0) = -1;
    M(2,1) = -3;
    M(2,2) =  5;
    M(2,3) =  5;
    M(3,0) = -2;
    M(3,1) = -4;
    M(3,2) = -5;
    M(3,3) =  6;

    // default: all connections are strong
    {
        SparseMatrix A = M;
        SparseMatrix S;
        cusp::precond::aggregation::symmetric_strength_of_connection(A, S);
        Matrix result = S;
        ASSERT_EQUAL(result == M, true);
    }

    // theta = 0.0: all connections are strong
    {
        SparseMatrix A = M;
        SparseMatrix S;
        cusp::precond::aggregation::symmetric_strength_of_connection(A, S, 0.0);
        Matrix result = S;
        ASSERT_EQUAL(result == M, true);
    }

    // theta = 0.5
    {
        SparseMatrix A = M;
        SparseMatrix S;
        cusp::precond::aggregation::symmetric_strength_of_connection(A, S, 0.5);
        Matrix result = S;

        // expected output
        Matrix N(4,4);
        N(0,0) =  3;
        N(0,1) =  0;
        N(0,2) =  0;
        N(0,3) =  0;
        N(1,0) =  0;
        N(1,1) =  4;
        N(1,2) =  3;
        N(1,3) =  4;
        N(2,0) =  0;
        N(2,1) = -3;
        N(2,2) =  5;
        N(2,3) =  5;
        N(3,0) =  0;
        N(3,1) = -4;
        N(3,2) = -5;
        N(3,3) =  6;
        ASSERT_EQUAL(result == N, true);
    }

    // theta = 0.75
    {
        SparseMatrix A = M;
        SparseMatrix S;
        cusp::precond::aggregation::symmetric_strength_of_connection(A, S, 0.75);
        Matrix result = S;

        // expected output
        Matrix N(4,4);
        N(0,0) =  3;
        N(0,1) =  0;
        N(0,2) =  0;
        N(0,3) =  0;
        N(1,0) =  0;
        N(1,1) =  4;
        N(1,2) =  0;
        N(1,3) =  4;
        N(2,0) =  0;
        N(2,1) =  0;
        N(2,2) =  5;
        N(2,3) =  5;
        N(3,0) =  0;
        N(3,1) = -4;
        N(3,2) = -5;
        N(3,3) =  6;
        ASSERT_EQUAL(result == N, true);
    }

    // theta = 0.9
    {
        SparseMatrix A = M;
        SparseMatrix S;
        cusp::precond::aggregation::symmetric_strength_of_connection(A, S, 0.9);
        Matrix result = S;

        // expected output
        Matrix N(4,4);
        N(0,0) =  3;
        N(0,1) =  0;
        N(0,2) =  0;
        N(0,3) =  0;
        N(1,0) =  0;
        N(1,1) =  4;
        N(1,2) =  0;
        N(1,3) =  0;
        N(2,0) =  0;
        N(2,1) =  0;
        N(2,2) =  5;
        N(2,3) =  5;
        N(3,0) =  0;
        N(3,1) =  0;
        N(3,2) = -5;
        N(3,3) =  6;
        ASSERT_EQUAL(result == N, true);
    }

}
DECLARE_SPARSE_MATRIX_UNITTEST(TestSymmetricStrengthOfConnection);

