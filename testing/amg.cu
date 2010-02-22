#include <unittest/unittest.h>

#include <cusp/array2d.h>

// REMOVE THIS
#include <cusp/print.h>
#include <cusp/gallery/poisson.h>
#include <cusp/io/matrix_market.h>


// TAKE THESE
#include <cusp/blas.h>
#include <cusp/multiply.h>
#include <cusp/transpose.h>
#include <cusp/relaxation/jacobi.h>
#include <cusp/graph/maximal_independent_set.h>
#include <cusp/detail/lu.h>

#include <thrust/count.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/transform.h>

#include <list>


template <typename IndexType>
struct filter_strong_connections
{
    template <typename Tuple>
    __host__ __device__
    IndexType operator()(const Tuple& t)
    {
        IndexType s_i = thrust::get<2>(t);
        IndexType s_j = thrust::get<3>(t);

        if (!s_i &&  s_j) return 1; // F->C connection
        if (!s_i && !s_j) return 0; // F->F connection

        IndexType i   = thrust::get<0>(t);
        IndexType j   = thrust::get<1>(t);
        
        if (s_i && i == j) return 1; // C->C connection (self connection)
        else return 0;
    }
};

template <typename IndexType, typename ValueType>
struct is_F_node : public thrust::unary_function<IndexType,ValueType>
{
    __host__ __device__
    ValueType operator()(const IndexType& i) const
    {
        return (i) ? ValueType(0) : ValueType(1);
    }
};

template <typename ValueType>
struct compute_weights
{
    template <typename Tuple>
    __host__ __device__
    ValueType operator()(const Tuple& t, const ValueType& v)
    {
        if (thrust::get<0>(t))  // C node w_ij = 0
            return 1;
        else                    // F node w_ij = |A_ij| / nu
            return ((v < 0) ? -v : v) / thrust::get<1>(t);
    }
};

template <typename IndexType, typename ValueType, typename MemorySpace,
          typename ArrayType>
void direct_interpolation(const cusp::coo_matrix<IndexType,ValueType,MemorySpace>& A,
                          const cusp::coo_matrix<IndexType,ValueType,MemorySpace>& C,
                          const ArrayType& cf_splitting,                              
                          cusp::coo_matrix<IndexType,ValueType,MemorySpace>& P)
{
    // dimensions of P
    const IndexType num_rows = A.num_rows;
    const IndexType num_cols = thrust::count(cf_splitting.begin(), cf_splitting.end(), 1);
  
    // mark the strong edges that are retained in P (either F->C or C->C self loops)
    cusp::array1d<IndexType,MemorySpace> stencil(C.num_entries);
    thrust::transform(thrust::make_zip_iterator(
                        thrust::make_tuple(C.row_indices.begin(),
                                           C.column_indices.begin(),
                                           thrust::make_permutation_iterator(cf_splitting.begin(), C.row_indices.begin()),
                                           thrust::make_permutation_iterator(cf_splitting.begin(), C.column_indices.begin()))),
                      thrust::make_zip_iterator(
                        thrust::make_tuple(C.row_indices.begin(),
                                           C.column_indices.begin(),
                                           thrust::make_permutation_iterator(cf_splitting.begin(), C.row_indices.begin()),
                                           thrust::make_permutation_iterator(cf_splitting.begin(), C.column_indices.begin()))) + C.num_entries,
                      stencil.begin(),
                      filter_strong_connections<IndexType>());

    // number of entries in P (number of F->C connections plus the number of C nodes)
    const IndexType num_entries = thrust::reduce(stencil.begin(), stencil.end());

    // sum the weights of the F nodes within each row
    cusp::array1d<ValueType,MemorySpace> nu(A.num_rows);
    {
        // nu = 1 / A * [F0F0F0]
        // scale C(i,j) by nu
        cusp::array1d<ValueType,MemorySpace> F_nodes(A.num_rows);  // 1.0 for F nodes, 0.0 for C nodes
        thrust::transform(cf_splitting.begin(), cf_splitting.end(), F_nodes.begin(), is_F_node<IndexType,ValueType>());
        cusp::multiply(A, F_nodes, nu);
    }
    
    // allocate storage for P
    {
        cusp::coo_matrix<IndexType,ValueType,MemorySpace> temp(num_rows, num_cols, num_entries);
        P.swap(temp);
    }

    // compute entries of P
    {
        // enumerate the C nodes
        cusp::array1d<ValueType,MemorySpace> coarse_index_map(A.num_rows);
        thrust::exclusive_scan(cf_splitting.begin(), cf_splitting.end(), coarse_index_map.begin());
       
        // TODO merge these copy_if() with a zip_iterator
        thrust::copy_if(C.row_indices.begin(), C.row_indices.end(),
                        stencil.begin(),
                        P.row_indices.begin(),
                        thrust::identity<IndexType>());
        thrust::copy_if(thrust::make_permutation_iterator(coarse_index_map.begin(), C.column_indices.begin()),
                        thrust::make_permutation_iterator(coarse_index_map.begin(), C.column_indices.end()),
                        stencil.begin(),
                        P.column_indices.begin(),
                        thrust::identity<IndexType>());
        thrust::copy_if(C.values.begin(), C.values.end(),
                        stencil.begin(),
                        P.values.begin(),
                        thrust::identity<IndexType>());

        thrust::transform(thrust::make_permutation_iterator(thrust::make_zip_iterator(thrust::make_tuple(cf_splitting.begin(), nu.begin())), P.row_indices.begin()), 
                          thrust::make_permutation_iterator(thrust::make_zip_iterator(thrust::make_tuple(cf_splitting.begin(), nu.begin())), P.row_indices.end()), 
                          P.values.begin(),
                          P.values.begin(),
                          compute_weights<ValueType>());
    }
}

template <typename IndexType, typename ValueType, typename Space>
void _TestDirectInterpolation(const cusp::array2d<ValueType,Space>& A,
                              const cusp::array1d<IndexType,Space>& S,
                              const cusp::array2d<ValueType,Space>& expected)
{
    cusp::coo_matrix<IndexType,ValueType,Space> A_(A);
    
    cusp::coo_matrix<IndexType,ValueType,Space> P;

    direct_interpolation(A_, A_, S, P);

    cusp::array2d<ValueType, Space> result(P);

    ASSERT_EQUAL_QUIET(result, expected);
}

template <class Space>
void TestDirectInterpolation(void)
{
    // One-dimensional Poisson problem 
    {
        cusp::array2d<float, Space> A(5,5);
        A(0,0) =  2;  A(0,1) = -1;  A(0,2) =  0;  A(0,3) =  0;  A(0,4) =  0; 
        A(1,0) = -1;  A(1,1) =  2;  A(1,2) = -1;  A(1,3) =  0;  A(1,4) =  0;
        A(2,0) =  0;  A(2,1) = -1;  A(2,2) =  2;  A(2,3) = -1;  A(2,4) =  0;
        A(3,0) =  0;  A(3,1) =  0;  A(3,2) = -1;  A(3,3) =  2;  A(3,4) = -1;
        A(4,0) =  0;  A(4,1) =  0;  A(4,2) =  0;  A(4,3) = -1;  A(4,4) =  2;

        cusp::array1d<int, Space> S(5);
        S[0] = 1;
        S[1] = 0;
        S[2] = 1;
        S[3] = 0;
        S[4] = 1;

        cusp::array2d<float, Space> P(5, 3);
        P(0,0) = 1.0;  P(0,1) = 0.0;  P(0,2) = 0.0;
        P(1,0) = 0.5;  P(1,1) = 0.5;  P(1,2) = 0.0;
        P(2,0) = 0.0;  P(2,1) = 1.0;  P(2,2) = 0.0;
        P(3,0) = 0.0;  P(3,1) = 0.5;  P(3,2) = 0.5;
        P(4,0) = 0.0;  P(4,1) = 0.0;  P(4,2) = 1.0;

        _TestDirectInterpolation(A,S,P);
    }
    
    // Two-dimensional Poisson problem 
    {
        cusp::array2d<float, Space> A(6,6);
        A(0,0) =  4;  A(0,1) = -1;  A(0,2) =  0;  A(0,3) = -1;  A(0,4) =  0;  A(0,5) =  0;  
        A(1,0) = -1;  A(1,1) =  4;  A(1,2) = -1;  A(1,3) =  0;  A(1,4) = -1;  A(1,5) =  0;
        A(2,0) =  0;  A(2,1) = -1;  A(2,2) =  4;  A(2,3) =  0;  A(2,4) =  0;  A(2,5) = -1;
        A(3,0) = -1;  A(3,1) =  0;  A(3,2) =  0;  A(3,3) =  4;  A(3,4) = -1;  A(3,5) =  0;
        A(4,0) =  0;  A(4,1) = -1;  A(4,2) =  0;  A(4,3) = -1;  A(4,4) =  4;  A(4,5) = -1;
        A(5,0) =  0;  A(5,1) =  0;  A(5,2) = -1;  A(5,3) =  0;  A(5,4) = -1;  A(5,5) =  4;

        cusp::array1d<int, Space> S(6);
        S[0] = 1;
        S[1] = 0;
        S[2] = 1;
        S[3] = 0;
        S[4] = 1;
        S[5] = 0;

        cusp::array2d<float, Space> P(6, 3);
        P(0,0) = 1.00;  P(0,1) = 0.00;  P(0,2) = 0.00;
        P(1,0) = 0.25;  P(1,1) = 0.25;  P(1,2) = 0.25;
        P(2,0) = 0.00;  P(2,1) = 1.00;  P(2,2) = 0.00;
        P(3,0) = 0.25;  P(3,1) = 0.00;  P(3,2) = 0.25;
        P(4,0) = 0.00;  P(4,1) = 0.00;  P(4,2) = 1.00;
        P(5,0) = 0.00;  P(5,1) = 0.25;  P(5,2) = 0.25;

        _TestDirectInterpolation(A,S,P);
    }
}
DECLARE_HOST_DEVICE_UNITTEST(TestDirectInterpolation);

template <typename IndexType, typename ValueType, typename MemorySpace>
class ruge_stuben_solver
{
    
    struct level
    {
        cusp::coo_matrix<IndexType,ValueType,MemorySpace> R;  // restriction operator
        cusp::coo_matrix<IndexType,ValueType,MemorySpace> A;  // matrix
        cusp::coo_matrix<IndexType,ValueType,MemorySpace> P;  // prolongation operator
        cusp::array1d<int,MemorySpace> splitting;             // C/F splitting
        
        cusp::relaxation::jacobi<ValueType,MemorySpace> smoother;
       
        ValueType rho;                                        // spectral radius
        //cusp::array1d<ValueType,MemorySpace> temp1;
        //cusp::array1d<ValueType,MemorySpace> temp2;
    };

    std::list<level> levels;
        
    cusp::detail::lu_solver<float, cusp::host_memory> LU;

    public:

    ruge_stuben_solver(const cusp::coo_matrix<IndexType,ValueType,MemorySpace>& A)
    {
        levels.push_back(level());
        levels.back().A = A; // copy

        extend_hierarchy();
   
        // TODO make lu_solver accept sparse input
        cusp::array2d<float,cusp::host_memory> coarse_dense(levels.back().A);
        LU = cusp::detail::lu_solver<float, cusp::host_memory>(coarse_dense);
    }

    void extend_hierarchy(void)
    {
        const cusp::coo_matrix<IndexType,ValueType,MemorySpace>& A = levels.back().A;

        // compute C/F splitting
        cusp::array1d<int,cusp::device_memory> splitting(A.num_rows);
        {
            // TODO implement MIS for coo_matrix
            cusp::csr_matrix<int,float,cusp::device_memory> csr(A);
            cusp::graph::maximal_independent_set(csr, splitting);
        }

        //    // TODO XXX XXX XXX remove
        //    for(int i = 0; i < splitting.size(); i++)
        //        splitting[i] = (i + 1) % 2;

        // compute prolongation operator
        cusp::coo_matrix<int,float,cusp::device_memory> P;
        direct_interpolation(A, A, splitting, P);

        // compute restriction operator (transpose of prolongator)
        cusp::coo_matrix<int,float,cusp::device_memory> R;
        cusp::transpose(P,R);

        // construct Galerkin product R*A*P
        cusp::coo_matrix<int,float,cusp::device_memory> RAP;
        {
            // TODO test speed of R * (A * P) vs. (R * A) * P
            cusp::coo_matrix<int,float,cusp::device_memory> AP;
            cusp::multiply(A, P, AP);
            cusp::multiply(R, AP, RAP);
        }

        levels.back().R.swap(R);
        levels.back().P.swap(P);
        levels.back().smoother = cusp::relaxation::jacobi<ValueType, MemorySpace>(A, 0.66f);  
        levels.back().splitting.swap(splitting);

        levels.push_back(level());
        levels.back().A.swap(RAP);
        //levels.back().temp1.resize(RAP.num_rows);
        //levels.back().temp2.resize(RAP.num_rows);
    }
    
    void solve(const cusp::array1d<float,cusp::device_memory> b,
                     cusp::array1d<float,cusp::device_memory> x)
    {
        // TODO check sizes
        cusp::coo_matrix<int,float,cusp::device_memory> & R = levels.front().R;
        cusp::coo_matrix<int,float,cusp::device_memory> & A = levels.front().A;
        cusp::coo_matrix<int,float,cusp::device_memory> & P = levels.front().P;
        
        cusp::relaxation::jacobi<ValueType, MemorySpace>& smoother = levels.front().smoother;
        
        cusp::coo_matrix<int,float,cusp::device_memory> & RAP = levels.back().A;

        //  4/3 * 1/rho is a good default, where rho is the spectral radius of D^-1(A)

        cusp::array1d<float,cusp::device_memory> residual(A.num_rows);
        cusp::array1d<float,cusp::device_memory> coarse(RAP.num_rows);
            
        // compute initial residual norm
        cusp::multiply(A,x,residual);
        cusp::blas::axpby(b, residual, residual, 1.0f, -1.0f);
        float last_norm = cusp::blas::nrm2(residual);

        // perform V-cycles on 2-level hierarchy
        for (unsigned int i = 0; i < 25; i++)
        {
            // presmooth
            smoother(A,b,x);

            // compute residual <- b - A*x
            cusp::multiply(A,x,residual);
            cusp::blas::axpby(b, residual, residual, 1.0f, -1.0f);

            // restrict to coarse grid
            cusp::multiply(R, residual, coarse);

            // compute coarse grid solution
            {
                // TODO streamline
                cusp::array1d<float,cusp::host_memory> temp_b(coarse);
                cusp::array1d<float,cusp::host_memory> temp_x(coarse.size());
                LU(temp_b, temp_x);
                coarse = temp_x;
            }

            // apply coarse grid correction 
            cusp::multiply(P, coarse, residual);
            cusp::blas::axpy(residual, x, 1.0f);

            // postsmooth
            smoother(A,b,x);

            float norm = cusp::blas::nrm2(x);

            //printf("%10.8f  %6.4f\n", norm, norm/last_norm);

            last_norm = norm;
        }
    }
        
};


void TestRugeStubenSolver(void)
{
    // Create 2D Poisson problem
    cusp::coo_matrix<int,float,cusp::device_memory> A;
    cusp::gallery::poisson5pt(A, 51, 51);
    
    // setup linear system
    cusp::array1d<float,cusp::device_memory> b(A.num_rows,0);
    cusp::array1d<float,cusp::device_memory> x(A.num_rows,1);
    unittest::random_samples(x.begin(), x.end());

    ruge_stuben_solver<int,float,cusp::device_memory> rs(A);
    rs.solve(b,x);
}
DECLARE_UNITTEST(TestRugeStubenSolver);

