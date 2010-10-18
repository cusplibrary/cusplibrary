#include <unittest/unittest.h>

#include <cusp/precond/smoothed_aggregation.h>

#include <cusp/gallery/poisson.h>
#include <cusp/krylov/cg.h>

template <class MemorySpace>
void TestStandardAggregation(void)
{
    // TODO make this test something, possibly disjoint things that must aggregate

    cusp::coo_matrix<int,float,MemorySpace> A;
    cusp::gallery::poisson5pt(A, 10, 10);

    cusp::array1d<int,MemorySpace> aggregates(A.num_rows);
    cusp::precond::detail::standard_aggregation(A, aggregates);
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
        ASSERT_EQUAL((std::abs(cusp::precond::detail::estimate_rho_Dinv_A(A) - rho) / rho) < 0.1f, true);
    }

    // 2x2 Poisson problem
    {
        cusp::csr_matrix<int, float, MemorySpace> A; cusp::gallery::poisson5pt(A, 2, 2); 
        float rho = 1.5;
        ASSERT_EQUAL((std::abs(cusp::precond::detail::estimate_rho_Dinv_A(A) - rho) / rho) < 0.1f, true);
    }

    // 4x4 Poisson problem
    {
        cusp::csr_matrix<int, float, MemorySpace> A; cusp::gallery::poisson5pt(A, 4, 4); 
        float rho = 1.8090169943749468;
        ASSERT_EQUAL((std::abs(cusp::precond::detail::estimate_rho_Dinv_A(A) - rho) / rho) < 0.1f, true);
    }
}
DECLARE_HOST_DEVICE_UNITTEST(TestEstimateRhoDinvA);


template <typename MemorySpace>
void TestFitCandidates(void)
{
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

        cusp::coo_matrix<int,float,MemorySpace> Q;
        cusp::array1d<float,MemorySpace> R(2);

        cusp::precond::detail::fit_candidates(aggregates, B, Q, R);

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

        cusp::coo_matrix<int,float,MemorySpace> Q;
        cusp::array1d<float,MemorySpace> R(4);
   
        cusp::precond::detail::fit_candidates(aggregates, B, Q, R);

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
    // simple example with diagonal S
    {
        cusp::coo_matrix<int, float, MemorySpace> S(4,4,4);
        S.row_indices[0] = 0; S.column_indices[0] = 0; S.values[0] = 1;
        S.row_indices[1] = 1; S.column_indices[1] = 1; S.values[1] = 2;
        S.row_indices[2] = 2; S.column_indices[2] = 2; S.values[2] = 3;
        S.row_indices[3] = 3; S.column_indices[3] = 3; S.values[3] = 4;

        cusp::coo_matrix<int, float, MemorySpace> T(4,2,4);
        T.row_indices[0] = 0; T.column_indices[0] = 0; T.values[0] = 0.5;
        T.row_indices[1] = 1; T.column_indices[1] = 0; T.values[1] = 0.5;
        T.row_indices[2] = 2; T.column_indices[2] = 1; T.values[2] = 0.5;
        T.row_indices[3] = 3; T.column_indices[3] = 1; T.values[3] = 0.5;

        cusp::coo_matrix<int, float, MemorySpace> P;

        cusp::precond::detail::smooth_prolongator(S, T, P, 4.0f, 2.0f); 

        ASSERT_EQUAL(P.num_rows,    4);
        ASSERT_EQUAL(P.num_cols,    2);
        ASSERT_EQUAL(P.num_entries, 4);
        
        ASSERT_EQUAL(P.row_indices[0], 0); ASSERT_EQUAL(P.column_indices[0], 0);
        ASSERT_EQUAL(P.row_indices[1], 1); ASSERT_EQUAL(P.column_indices[1], 0);
        ASSERT_EQUAL(P.row_indices[2], 2); ASSERT_EQUAL(P.column_indices[2], 1);
        ASSERT_EQUAL(P.row_indices[3], 3); ASSERT_EQUAL(P.column_indices[3], 1);

        ASSERT_EQUAL(P.values[0], -0.5);
        ASSERT_EQUAL(P.values[1], -0.5);
        ASSERT_EQUAL(P.values[2], -0.5);
        ASSERT_EQUAL(P.values[3], -0.5);
    }

    // 1D Poisson problem w/ 4 points and 2 aggregates
    {
        cusp::coo_matrix<int, float, MemorySpace> S(4,4,10);
        S.row_indices[0] = 0; S.column_indices[0] = 0; S.values[0] = 2;
        S.row_indices[1] = 0; S.column_indices[1] = 1; S.values[1] =-1;
        S.row_indices[2] = 1; S.column_indices[2] = 0; S.values[2] =-1;
        S.row_indices[3] = 1; S.column_indices[3] = 1; S.values[3] = 2;
        S.row_indices[4] = 1; S.column_indices[4] = 2; S.values[4] =-1;
        S.row_indices[5] = 2; S.column_indices[5] = 1; S.values[5] =-1;
        S.row_indices[6] = 2; S.column_indices[6] = 2; S.values[6] = 2;
        S.row_indices[7] = 2; S.column_indices[7] = 3; S.values[7] =-1;
        S.row_indices[8] = 3; S.column_indices[8] = 2; S.values[8] =-1;
        S.row_indices[9] = 3; S.column_indices[9] = 3; S.values[9] = 2;

        cusp::coo_matrix<int, float, MemorySpace> T(4,2,4);
        T.row_indices[0] = 0; T.column_indices[0] = 0; T.values[0] = 0.5;
        T.row_indices[1] = 1; T.column_indices[1] = 0; T.values[1] = 0.5;
        T.row_indices[2] = 2; T.column_indices[2] = 1; T.values[2] = 0.5;
        T.row_indices[3] = 3; T.column_indices[3] = 1; T.values[3] = 0.5;

        cusp::coo_matrix<int, float, MemorySpace> P;

        cusp::precond::detail::smooth_prolongator(S, T, P, 4.0f/3.0f, 1.8090169943749472f); 

        ASSERT_EQUAL(P.num_rows,    4);
        ASSERT_EQUAL(P.num_cols,    2);
        ASSERT_EQUAL(P.num_entries, 6);
        
        ASSERT_EQUAL(P.row_indices[0], 0); ASSERT_EQUAL(P.column_indices[0], 0);
        ASSERT_EQUAL(P.row_indices[1], 1); ASSERT_EQUAL(P.column_indices[1], 0);
        ASSERT_EQUAL(P.row_indices[2], 1); ASSERT_EQUAL(P.column_indices[2], 1);
        ASSERT_EQUAL(P.row_indices[3], 2); ASSERT_EQUAL(P.column_indices[3], 0);
        ASSERT_EQUAL(P.row_indices[4], 2); ASSERT_EQUAL(P.column_indices[4], 1);
        ASSERT_EQUAL(P.row_indices[5], 3); ASSERT_EQUAL(P.column_indices[5], 1);

        ASSERT_ALMOST_EQUAL(P.values[0], 0.31573787f);
        ASSERT_ALMOST_EQUAL(P.values[1], 0.31573787f);
        ASSERT_ALMOST_EQUAL(P.values[2], 0.18426213f);
        ASSERT_ALMOST_EQUAL(P.values[3], 0.18426213f);
        ASSERT_ALMOST_EQUAL(P.values[4], 0.31573787f);
        ASSERT_ALMOST_EQUAL(P.values[5], 0.31573787f);
    }
}
DECLARE_HOST_DEVICE_UNITTEST(TestSmoothProlongator);

template <class MemorySpace>
void TestSmoothedAggregation(void)
{
    typedef int                 IndexType;
    typedef float               ValueType;

    // Create 2D Poisson problem
    cusp::coo_matrix<IndexType,ValueType,MemorySpace> A;
    cusp::gallery::poisson5pt(A, 100, 100);
    
    // create smoothed aggregation solver
    cusp::precond::smoothed_aggregation<IndexType,ValueType,MemorySpace> M(A);

    // test as standalone solver
    {
        cusp::array1d<ValueType,MemorySpace> b = unittest::random_samples<ValueType>(A.num_rows);
        cusp::array1d<ValueType,MemorySpace> x = unittest::random_samples<ValueType>(A.num_rows);
    
        cusp::convergence_monitor<ValueType> monitor(b, 40, 1e-5);
        M.solve(b,x,monitor);

        ASSERT_EQUAL(monitor.converged(), true);
        ASSERT_EQUAL(monitor.geometric_rate() < 0.9, true);
    }

    // test as preconditioner
    {
        cusp::array1d<ValueType,MemorySpace> b = unittest::random_samples<ValueType>(A.num_rows);
        cusp::array1d<ValueType,MemorySpace> x = unittest::random_samples<ValueType>(A.num_rows);

        // set stopping criteria (iteration_limit = 20, relative_tolerance = 1e-5)
        cusp::default_monitor<ValueType> monitor(b, 20, 1e-5);
        cusp::krylov::cg(A, x, b, monitor, M);

        ASSERT_EQUAL(monitor.converged(), true);
    }
}
DECLARE_HOST_DEVICE_UNITTEST(TestSmoothedAggregation);

