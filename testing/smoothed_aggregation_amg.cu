#include <unittest/unittest.h>

#include <cusp/array2d.h>

// REMOVE THIS
#include <cusp/print.h>
#include <cusp/gallery/poisson.h>
#include <cusp/io/matrix_market.h>
#include <cusp/krylov/cg.h>

// TAKE THESE
#include <cusp/blas.h>
#include <cusp/multiply.h>
#include <cusp/transpose.h>
#include <cusp/relaxation/jacobi.h>
#include <cusp/graph/maximal_independent_set.h>
#include <cusp/precond/diagonal.h>
#include <cusp/detail/lu.h>
#include <cusp/detail/spectral_radius.h>

#include <thrust/extrema.h>
#include <thrust/transform.h>
#include <thrust/gather.h>
#include <thrust/reduce.h>

#include <vector>

#define CHECK_NAN(a)                                          \
{                                                             \
    cusp::array1d<ValueType,cusp::host_memory> h(a);          \
    for(size_t i = 0; i < h.size(); i++)                      \
        if (isnan(h[i]))                                      \
            printf("[%d] nan at index %d\n", __LINE__, (int) i);    \
}                    



template <typename IndexType, typename ValueType, typename MemorySpace,
          typename ArrayType>
void mis_to_aggregates(const cusp::coo_matrix<IndexType,ValueType,MemorySpace>& C,
                       const ArrayType& mis,
                             ArrayType& aggregates)
{
    // 
    const IndexType N = C.num_rows;

    // (2,i) mis (0,i) non-mis
    cusp::csr_matrix<IndexType,ValueType,MemorySpace> A(C);

    // current (ring,index)
    ArrayType mis1(N);
    ArrayType idx1(N);
    ArrayType mis2(N);
    ArrayType idx2(N);

    typedef typename ArrayType::value_type T;
    typedef thrust::tuple<T,T> Tuple;

    // find the largest (mis[j],j) 1-ring neighbor for each node
    cusp::detail::device::cuda::spmv_csr_scalar
        (A.num_rows,
         A.row_offsets.begin(), A.column_indices.begin(), thrust::constant_iterator<int>(1),  // XXX should we mask explicit zeros? (e.g. DIA, array2d)
         thrust::make_zip_iterator(thrust::make_tuple(mis.begin(), thrust::counting_iterator<IndexType>(0))),
         thrust::make_zip_iterator(thrust::make_tuple(mis.begin(), thrust::counting_iterator<IndexType>(0))),
         thrust::make_zip_iterator(thrust::make_tuple(mis1.begin(), idx1.begin())),
         thrust::identity<Tuple>(), thrust::project2nd<Tuple,Tuple>(), thrust::maximum<Tuple>());

    // boost mis0 values so they win in second round
    thrust::transform(mis.begin(), mis.end(), mis1.begin(), mis1.begin(), thrust::plus<typename ArrayType::value_type>());

    // find the largest (mis[j],j) 2-ring neighbor for each node
    cusp::detail::device::cuda::spmv_csr_scalar
        (A.num_rows,
         A.row_offsets.begin(), A.column_indices.begin(), thrust::constant_iterator<int>(1),  // XXX should we mask explicit zeros? (e.g. DIA, array2d)
         thrust::make_zip_iterator(thrust::make_tuple(mis1.begin(), idx1.begin())),
         thrust::make_zip_iterator(thrust::make_tuple(mis1.begin(), idx1.begin())),
         thrust::make_zip_iterator(thrust::make_tuple(mis2.begin(), idx2.begin())),
         thrust::identity<Tuple>(), thrust::project2nd<Tuple,Tuple>(), thrust::maximum<Tuple>());

    // enumerate the MIS nodes
    cusp::array1d<IndexType,MemorySpace> mis_enum(N);
    thrust::exclusive_scan(mis.begin(), mis.end(), mis_enum.begin());

    thrust::gather(idx2.begin(), idx2.end(),
                   mis_enum.begin(),
                   aggregates.begin());
}

template <typename IndexType, typename ValueType, typename MemorySpace,
          typename ArrayType>
void standard_aggregation(const cusp::coo_matrix<IndexType,ValueType,MemorySpace>& C,
                                ArrayType& aggregates)
{
    // TODO check sizes
    // TODO label singletons with a -1

    const size_t N = C.num_rows;

    cusp::array1d<IndexType,MemorySpace> mis(N);
    // compute MIS(2)
    {
        // TODO implement MIS for coo_matrix
        cusp::csr_matrix<IndexType,ValueType,MemorySpace> csr(C);
        cusp::graph::maximal_independent_set(csr, mis, 2);
    }

    mis_to_aggregates(C, mis, aggregates);
}

template <typename T>
struct square : thrust::unary_function<T,T>
{
    __host__ __device__
    T operator()(const T& x) { return x * x; }
};

template <typename T>
struct sqrt_functor : thrust::unary_function<T,T>
{
    __host__ __device__
    T operator()(const T& x) { return sqrt(x); }
};

template <typename Array1,
          typename Array2,
          typename IndexType, typename ValueType, typename MemorySpace,
          typename Array3>
void fit_candidates(const Array1& aggregates,
                    const Array2& B,
                          cusp::coo_matrix<IndexType,ValueType,MemorySpace>& Q,
                          Array3& R)
{
    // TODO handle case w/ unaggregated nodes (marked w/ -1)
    IndexType num_aggregates = *thrust::max_element(aggregates.begin(), aggregates.end()) + 1;

    Q.resize(aggregates.size(), num_aggregates, aggregates.size());
    R.resize(num_aggregates);

    // gather values into Q
    thrust::sequence(Q.row_indices.begin(), Q.row_indices.end());
    thrust::copy(aggregates.begin(), aggregates.end(), Q.column_indices.begin());
    thrust::copy(B.begin(), B.end(), Q.values.begin());
                          
    // compute norm over each aggregate
    {
        // compute Qt
        cusp::coo_matrix<IndexType,ValueType,MemorySpace> Qt;  cusp::transpose(Q, Qt);

        // compute sum of squares for each column of Q (rows of Qt)
        cusp::array1d<IndexType, MemorySpace> temp(num_aggregates);
        thrust::reduce_by_key(Qt.row_indices.begin(), Qt.row_indices.end(),
                              thrust::make_transform_iterator(Qt.values.begin(), square<ValueType>()),
                              temp.begin(),
                              R.begin());

        // compute square root of each column sum
        thrust::transform(R.begin(), R.end(), R.begin(), sqrt_functor<ValueType>());
    }

    // rescale columns of Q
    thrust::transform(Q.values.begin(), Q.values.end(),
                      thrust::make_permutation_iterator(R.begin(), Q.column_indices.begin()),
                      Q.values.begin(),
                      thrust::divides<ValueType>());
    
    //std::cout << "Q" << std::endl;
    //cusp::print_matrix(Q);

    //std::cout << "R" << std::endl;
    //cusp::print_matrix(R);
}


//   Smoothed (final) prolongator defined by P = (I - omega/rho(K) K) * T
//   where K = diag(S)^-1 * S and rho(K) is an approximation to the 
//   spectral radius of K.
template <typename IndexType, typename ValueType, typename MemorySpace>
void smooth_prolongator(const cusp::coo_matrix<IndexType,ValueType,MemorySpace>& S,
                        const cusp::coo_matrix<IndexType,ValueType,MemorySpace>& T,
                              cusp::coo_matrix<IndexType,ValueType,MemorySpace>& P,
                        const ValueType omega = 4.0/3.0,
                        const ValueType rho_Dinv_S = 0.0)
{
    // TODO handle case with unaggregated nodes
    assert(T.num_entries == T.num_rows);

    const ValueType lambda = omega / (rho_Dinv_S == 0.0 ? estimate_rho_Dinv_A(S) : rho_Dinv_S);

    // temp <- lambda * S(i,j) * T(j,k)
    cusp::coo_matrix<IndexType,ValueType,MemorySpace> temp(S.num_rows, T.num_cols, S.num_entries + T.num_entries);
    thrust::copy(S.row_indices.begin(), S.row_indices.end(), temp.row_indices.begin());
    thrust::gather(S.column_indices.begin(), S.column_indices.end(), T.column_indices.begin(), temp.column_indices.begin());
    thrust::transform(S.values.begin(), S.values.end(),
                      thrust::make_permutation_iterator(T.values.begin(), S.column_indices.begin()),
                      temp.values.begin(),
                      thrust::multiplies<ValueType>());
    thrust::transform(temp.values.begin(), temp.values.begin() + S.num_entries,
                      thrust::constant_iterator<ValueType>(-lambda),
                      temp.values.begin(),
                      thrust::multiplies<ValueType>());
    // temp <- D^-1
    {
        cusp::array1d<ValueType, MemorySpace> D(S.num_rows);
        cusp::detail::extract_diagonal(S, D);
        thrust::transform(temp.values.begin(), temp.values.begin() + S.num_entries,
                          thrust::make_permutation_iterator(D.begin(), S.row_indices.begin()),
                          temp.values.begin(),
                          thrust::divides<ValueType>());
    }

    // temp <- temp + T
    thrust::copy(T.row_indices.begin(),    T.row_indices.end(),    temp.row_indices.begin()    + S.num_entries);
    thrust::copy(T.column_indices.begin(), T.column_indices.end(), temp.column_indices.begin() + S.num_entries);
    thrust::copy(T.values.begin(),         T.values.end(),         temp.values.begin()         + S.num_entries);

    // sort by (I,J)
    {
        // TODO use explicit permuation and temporary arrays for efficiency
        thrust::sort_by_key(temp.column_indices.begin(), temp.column_indices.end(), thrust::make_zip_iterator(thrust::make_tuple(temp.row_indices.begin(),    temp.values.begin())));
        thrust::sort_by_key(temp.row_indices.begin(),    temp.row_indices.end(),    thrust::make_zip_iterator(thrust::make_tuple(temp.column_indices.begin(), temp.values.begin())));
    }


    // compute unique number of nonzeros in the output
    IndexType NNZ = thrust::inner_product(thrust::make_zip_iterator(thrust::make_tuple(temp.row_indices.begin(), temp.column_indices.begin())),
                                          thrust::make_zip_iterator(thrust::make_tuple(temp.row_indices.end (),  temp.column_indices.end()))   - 1,
                                          thrust::make_zip_iterator(thrust::make_tuple(temp.row_indices.begin(), temp.column_indices.begin())) + 1,
                                          IndexType(0),
                                          thrust::plus<IndexType>(),
                                          thrust::not_equal_to< thrust::tuple<IndexType,IndexType> >()) + 1;

    // allocate space for output
    P.resize(temp.num_rows, temp.num_cols, NNZ);

    // sum values with the same (i,j)
    thrust::reduce_by_key(thrust::make_zip_iterator(thrust::make_tuple(temp.row_indices.begin(), temp.column_indices.begin())),
                          thrust::make_zip_iterator(thrust::make_tuple(temp.row_indices.end(),   temp.column_indices.end())),
                          temp.values.begin(),
                          thrust::make_zip_iterator(thrust::make_tuple(P.row_indices.begin(), P.column_indices.begin())),
                          P.values.begin(),
                          thrust::equal_to< thrust::tuple<IndexType,IndexType> >(),
                          thrust::plus<ValueType>());

//    std::cout << "S" << std::endl;
//    cusp::print_matrix(S);
//
//    std::cout << "T" << std::endl;
//    cusp::print_matrix(T);
//
//    std::cout << "temp" << std::endl;
//    cusp::print_matrix(temp);
//    
//    std::cout << "P" << std::endl;
//    cusp::print_matrix(P);
}


template <typename MatrixType>
struct Dinv_A : public cusp::linear_operator<typename MatrixType::value_type, typename MatrixType::memory_space>
{
    const MatrixType& A;
    const cusp::precond::diagonal<typename MatrixType::value_type, typename MatrixType::memory_space> Dinv;

    Dinv_A(const MatrixType& A)
        : A(A), Dinv(A),
          cusp::linear_operator<typename MatrixType::value_type, typename MatrixType::memory_space>(A.num_rows, A.num_cols, A.num_entries + A.num_rows)
          {}

    template <typename Array1, typename Array2>
    void operator()(const Array1& x, Array2& y) const
    {
        cusp::multiply(A,x,y);
        cusp::multiply(Dinv,y,y);
    }
};

template <typename MatrixType>
double estimate_rho_Dinv_A(const MatrixType& A)
{
    typedef typename MatrixType::value_type   ValueType;
    typedef typename MatrixType::memory_space MemorySpace;

    Dinv_A<MatrixType> Dinv_A(A);

    return cusp::detail::estimate_spectral_radius(Dinv_A);
}


void TestStandardAggregation(void)
{
    cusp::coo_matrix<int,float,cusp::device_memory> A;
    cusp::gallery::poisson5pt(A, 10, 10);

    cusp::array1d<int,cusp::device_memory> aggregates(A.num_rows);
    standard_aggregation(A, aggregates);

//    cusp::print_matrix(aggregates);
}
DECLARE_UNITTEST(TestStandardAggregation);


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
        ASSERT_EQUAL((std::abs(estimate_rho_Dinv_A(A) - rho) / rho) < 0.1f, true);
    }

    // 2x2 Poisson problem
    {
        cusp::csr_matrix<int, float, MemorySpace> A; cusp::gallery::poisson5pt(A, 2, 2); 
        float rho = 1.5;
        ASSERT_EQUAL((std::abs(estimate_rho_Dinv_A(A) - rho) / rho) < 0.1f, true);
    }

    // 4x4 Poisson problem
    {
        cusp::csr_matrix<int, float, MemorySpace> A; cusp::gallery::poisson5pt(A, 4, 4); 
        float rho = 1.8090169943749468;
        ASSERT_EQUAL((std::abs(estimate_rho_Dinv_A(A) - rho) / rho) < 0.1f, true);
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

        fit_candidates(aggregates, B, Q, R);

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
   
        fit_candidates(aggregates, B, Q, R);

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

        smooth_prolongator(S, T, P, 4.0f, 2.0f); 

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

        smooth_prolongator(S, T, P, 4.0f/3.0f, 1.8090169943749472f); 

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

template <typename IndexType, typename ValueType, typename MemorySpace>
class smoothed_aggregation_solver: public cusp::linear_operator<ValueType, MemorySpace, IndexType>
{
    struct level
    {
        cusp::coo_matrix<IndexType,ValueType,MemorySpace> R;  // restriction operator
        cusp::coo_matrix<IndexType,ValueType,MemorySpace> A;  // matrix
        cusp::coo_matrix<IndexType,ValueType,MemorySpace> P;  // prolongation operator
        cusp::array1d<IndexType,MemorySpace> aggregates;      // aggregates
        cusp::array1d<ValueType,MemorySpace> B;               // near-nullspace candidates
        
        cusp::relaxation::jacobi<ValueType,MemorySpace> smoother;
       
        ValueType rho;                                        // spectral radius
    };

    std::vector<level> levels;
        
    cusp::detail::lu_solver<ValueType, cusp::host_memory> LU;

    public:

    smoothed_aggregation_solver(const cusp::coo_matrix<IndexType,ValueType,MemorySpace>& A)
    {
        levels.reserve(20); // avoid reallocations which force matrix copies

        levels.push_back(level());
        levels.back().A = A; // copy
        levels.back().B.resize(A.num_rows, 1.0f);

        extend_hierarchy();
        //extend_hierarchy();
        //extend_hierarchy();
   
        // TODO make lu_solver accept sparse input
        cusp::array2d<ValueType,cusp::host_memory> coarse_dense(levels.back().A);
        LU = cusp::detail::lu_solver<ValueType, cusp::host_memory>(coarse_dense);

        //printf("\n");
        //for (int i = 0; i < levels.size(); i++)
        //    printf("level[%2d] %10d unknowns %10d nonzeros\n", i, levels[i].A.num_rows, levels[i].A.num_entries);

        //{
        //    cusp::array2d<ValueType, MemorySpace> aggregates0(levels[0].aggregates.size(),1);
        //    aggregates0.values = levels[0].aggregates;
        //    cusp::io::write_matrix_market_file(aggregates0, "/home/nathan/Desktop/AMG/aggregates0.mtx");
        //}
        //cusp::io::write_matrix_market_file(levels[0].R, "/home/nathan/Desktop/AMG/R0.mtx");
        //cusp::io::write_matrix_market_file(levels[0].A, "/home/nathan/Desktop/AMG/A0.mtx");
        //cusp::io::write_matrix_market_file(levels[0].P, "/home/nathan/Desktop/AMG/P0.mtx");
        //cusp::io::write_matrix_market_file(levels[1].A, "/home/nathan/Desktop/AMG/A1.mtx");
        //cusp::io::write_matrix_market_file(levels[0].R, "/home/nathan/Desktop/AMG/aggregates1.mtx");
        //cusp::io::write_matrix_market_file(levels[1].R, "/home/nathan/Desktop/AMG/R1.mtx");
        //cusp::io::write_matrix_market_file(levels[1].P, "/home/nathan/Desktop/AMG/P1.mtx");
        //cusp::io::write_matrix_market_file(levels[2].A, "/home/nathan/Desktop/AMG/A2.mtx");
    }

    void extend_hierarchy(void)
    {
        const cusp::coo_matrix<IndexType,ValueType,MemorySpace>& A = levels.back().A;
        const cusp::array1d<ValueType,MemorySpace>&              B = levels.back().B;

        // compute stength of connection matrix
        const cusp::coo_matrix<IndexType,ValueType,MemorySpace>& C = A;
        // TODO add symmetric_strength of connection

        // compute spectral radius of diag(C)^-1 * C
        ValueType rho_DinvA = estimate_rho_Dinv_A(A);

        // compute aggregates
        cusp::array1d<IndexType,MemorySpace> aggregates(A.num_rows);
        standard_aggregation(A, aggregates);
        
        // compute tenative prolongator and coarse nullspace vector
        cusp::coo_matrix<IndexType,ValueType,MemorySpace> T;
        cusp::array1d<ValueType,MemorySpace>              B_coarse;
        fit_candidates(aggregates, B, T, B_coarse);
        
        //cusp::io::write_matrix_market_file(T, "/home/nathan/Desktop/AMG/T0.mtx");

        // compute prolongation operator
        cusp::coo_matrix<IndexType,ValueType,MemorySpace> P;
        smooth_prolongator(C, T, P, (ValueType) (4.0/3.0), rho_DinvA);  // TODO if C != A then compute rho_Dinv_C

        // compute restriction operator (transpose of prolongator)
        cusp::coo_matrix<IndexType,ValueType,MemorySpace> R;
        cusp::transpose(P,R);

        // construct Galerkin product R*A*P
        cusp::coo_matrix<IndexType,ValueType,MemorySpace> RAP;
        {
            // TODO test speed of R * (A * P) vs. (R * A) * P
            cusp::coo_matrix<IndexType,ValueType,MemorySpace> AP;
            cusp::multiply(A, P, AP);
            cusp::multiply(R, AP, RAP);
        }
        
        //  4/3 * 1/rho is a good default, where rho is the spectral radius of D^-1(A)
        ValueType omega = (4.0f/3.0f) / rho_DinvA;
        levels.back().smoother = cusp::relaxation::jacobi<ValueType, MemorySpace>(A, omega);
        levels.back().aggregates.swap(aggregates);
        levels.back().R.swap(R);
        levels.back().P.swap(P);

        //std::cout << "omega " << omega << std::endl;

        levels.push_back(level());
        levels.back().A.swap(RAP);
        levels.back().B.swap(B_coarse);
    }
    
    template <typename Array1, typename Array2>
    void operator()(const Array1& x, Array2& y) const
    {
        // perform 1 V-cycle
        _solve(x, y, 0);
    }

    void solve(const cusp::array1d<ValueType,cusp::device_memory>& b,
                     cusp::array1d<ValueType,cusp::device_memory>& x) const
    {
        // TODO check sizes
        const cusp::coo_matrix<IndexType,ValueType,MemorySpace> & A = levels[0].A;

        cusp::array1d<ValueType,MemorySpace> residual(A.num_rows);  // TODO eliminate temporaries
            
        // compute initial residual norm
        cusp::multiply(A,x,residual);
        cusp::blas::axpby(b, residual, residual, 1.0f, -1.0f);
        ValueType last_norm = cusp::blas::nrm2(residual);
            
        //printf("%10.8f\n", last_norm);

        // perform 25 V-cycles
        for (int i = 0; i < 25; i++)
        {
            _solve(b, x, 0);

            // compute residual norm
            cusp::multiply(A,x,residual);
            cusp::blas::axpby(b, residual, residual, 1.0f, -1.0f);
            ValueType norm = cusp::blas::nrm2(residual);

            //printf("%10.8f  %6.4f\n", norm, norm/last_norm);

            last_norm = norm;
        }
    }

    void _solve(const cusp::array1d<ValueType,MemorySpace>& b,
                      cusp::array1d<ValueType,MemorySpace>& x,
                const int i) const
    {
        if (i + 1 == levels.size())
        {
            // coarse grid solve
            // TODO streamline
            cusp::array1d<ValueType,cusp::host_memory> temp_b(b);
            cusp::array1d<ValueType,cusp::host_memory> temp_x(x.size());
            LU(temp_b, temp_x);
            x = temp_x;
        }
        else
        {
            const cusp::coo_matrix<IndexType,ValueType,MemorySpace> & R = levels[i].R;
            const cusp::coo_matrix<IndexType,ValueType,MemorySpace> & A = levels[i].A;
            const cusp::coo_matrix<IndexType,ValueType,MemorySpace> & P = levels[i].P;

            cusp::array1d<ValueType,MemorySpace> residual(P.num_rows);  // TODO eliminate temporaries
            cusp::array1d<ValueType,MemorySpace> coarse_b(P.num_cols);
            cusp::array1d<ValueType,MemorySpace> coarse_x(P.num_cols);

            // presmooth
            levels[i].smoother(A,b,x);

            // compute residual <- b - A*x
            cusp::multiply(A, x, residual);
            cusp::blas::axpby(b, residual, residual, 1.0f, -1.0f);

            // restrict to coarse grid
            cusp::multiply(R, residual, coarse_b);

            // compute coarse grid solution
            _solve(coarse_b, coarse_x, i + 1);

            // apply coarse grid correction 
            cusp::multiply(P, coarse_x, residual);
            cusp::blas::axpy(residual, x, 1.0f);

            // postsmooth
            levels[i].smoother(A,b,x);
        }
    }
};


void TestSmoothedAggregationSolver(void)
{
    typedef int                 IndexType;
    typedef float               ValueType;
    typedef cusp::device_memory MemorySpace;

    // Create 2D Poisson problem
    cusp::coo_matrix<IndexType,ValueType,MemorySpace> A;
//    cusp::gallery::poisson5pt(A, 4, 4);
    cusp::gallery::poisson5pt(A, 50, 50);
    
    // setup linear system
    cusp::array1d<ValueType,MemorySpace> b(A.num_rows,0);
    cusp::array1d<ValueType,MemorySpace> x = unittest::random_samples<ValueType>(A.num_rows);

    smoothed_aggregation_solver<IndexType,ValueType,MemorySpace> M(A);
    M.solve(b,x);

    // set stopping criteria (iteration_limit = 100, relative_tolerance = 1e-6)
    //cusp::verbose_monitor<ValueType> monitor(b, 100, 1e-6);
    //cusp::krylov::cg(A, x, b, monitor);//, M);
}
DECLARE_UNITTEST(TestSmoothedAggregationSolver);

