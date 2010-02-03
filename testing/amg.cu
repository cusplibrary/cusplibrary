#include <unittest/unittest.h>

#include <cusp/array2d.h>

// REMOVE THIS
#include <cusp/print.h>

// TAKE THESE
#include <cusp/multiply.h>

#include <thrust/count.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/transform.h>



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

template <typename IndexType, typename ValueType, typename SpaceOrAlloc,
          typename ArrayType>
void direct_interpolation(const cusp::coo_matrix<IndexType,ValueType,SpaceOrAlloc>& A,
                          const cusp::coo_matrix<IndexType,ValueType,SpaceOrAlloc>& C,
                          const ArrayType& cf_splitting,                              
                          cusp::coo_matrix<IndexType,ValueType,SpaceOrAlloc>& P)
{
    // dimensions of P
    const IndexType num_rows = A.num_rows;
    const IndexType num_cols = thrust::count(cf_splitting.begin(), cf_splitting.end(), 1);
  
    // mark the strong edges that are retained in P (either F->C or C->C self loops)
    cusp::array1d<IndexType,SpaceOrAlloc> stencil(C.num_entries);
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
    cusp::array1d<ValueType,SpaceOrAlloc> nu(A.num_rows);
    {
        // nu = 1 / A * [F0F0F0]
        // scale C(i,j) by nu
        cusp::array1d<ValueType,SpaceOrAlloc> F_nodes(A.num_rows);  // 1.0 for F nodes, 0.0 for C nodes
        thrust::transform(cf_splitting.begin(), cf_splitting.end(), F_nodes.begin(), is_F_node<IndexType,ValueType>());
        cusp::multiply(A, F_nodes, nu);
    }
    
    // allocate storage for P
    {
        cusp::coo_matrix<IndexType,ValueType,SpaceOrAlloc> temp(num_rows, num_cols, num_entries);
        P.swap(temp);
    }

    // compute entries of P
    {
        // enumerate the C nodes
        cusp::array1d<ValueType,SpaceOrAlloc> coarse_index_map(A.num_rows);
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

