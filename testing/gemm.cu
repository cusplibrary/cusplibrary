#include <unittest/unittest.h>
#include <cusp/coo_matrix.h>

#include <cusp/print.h>

#include <cusp/detail/format_utils.h>

#include <thrust/gather.h>
#include <thrust/scan.h>
#include <thrust/segmented_scan.h>
#include <thrust/scatter.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/unique.h>

template <typename IndexType, typename ValueType, typename SpaceOrAlloc>
void matrix_multiply(const cusp::coo_matrix<IndexType,ValueType,SpaceOrAlloc>& A,
                     const cusp::coo_matrix<IndexType,ValueType,SpaceOrAlloc>& B,
                           cusp::coo_matrix<IndexType,ValueType,SpaceOrAlloc>& C)
{
    typedef typename cusp::coo_matrix<IndexType,ValueType,SpaceOrAlloc>::memory_space MemorySpace;

    // check whether matrices are empty
    if (A.num_entries == 0 || B.num_entries == 0)
    {
        cusp::coo_matrix<IndexType,ValueType,SpaceOrAlloc> temp(A.num_rows, B.num_cols, 0);
        C.swap(temp);
        return;
    }

    // compute row offsets for B
    cusp::array1d<IndexType,MemorySpace> B_row_offsets(B.num_rows + 1);
    cusp::detail::indices_to_offsets(B.row_indices, B_row_offsets);

    // compute row lengths for B
    cusp::array1d<IndexType,MemorySpace> B_row_lengths(B.num_rows);
    thrust::transform(B_row_offsets.begin() + 1, B_row_offsets.end(), B_row_offsets.begin(), B_row_lengths.begin(), thrust::minus<IndexType>());

    // for each element A(i,j) compute the number of nonzero elements in B(j,:)
    cusp::array1d<IndexType,MemorySpace> segment_lengths(A.num_entries);
    thrust::gather(segment_lengths.begin(), segment_lengths.end(),
                   A.column_indices.begin(),
                   B_row_lengths.begin());
    
    // output pointer
    cusp::array1d<IndexType,MemorySpace> output_ptr(A.num_entries + 1);
    thrust::exclusive_scan(segment_lengths.begin(), segment_lengths.end(),
                           output_ptr.begin(),
                           IndexType(0));
    output_ptr[A.num_entries] = output_ptr[A.num_entries - 1] + segment_lengths[A.num_entries - 1]; // XXX is this necessary?

    IndexType coo_num_nonzeros = output_ptr[A.num_entries];
    
    // enumerate the segments in the intermediate format corresponding to each entry A(i,j)
    // XXX could be done with offset_to_index instead
    cusp::array1d<IndexType,MemorySpace> segments(coo_num_nonzeros, 0);
    thrust::scatter_if(thrust::counting_iterator<IndexType>(0), thrust::counting_iterator<IndexType>(A.num_entries),
                       output_ptr.begin(), 
                       segment_lengths.begin(),
                       segments.begin());
    thrust::inclusive_scan(segments.begin(), segments.end(), segments.begin(), thrust::maximum<IndexType>());
   
    // compute gather locations of intermediate format
    cusp::array1d<IndexType,MemorySpace> gather_locations(coo_num_nonzeros, 1);
    {
        // TODO replace temp arrays with permutation_iterator
        // TODO fuse two calls to scatter_if with zip_iterator
        cusp::array1d<IndexType,MemorySpace> temp(A.num_entries);  // B_row_offsets[Aj[n]]

        thrust::gather(temp.begin(), temp.end(),
                       A.column_indices.begin(),    
                       B_row_offsets.begin());

        thrust::scatter_if(temp.begin(), temp.end(),
                           output_ptr.begin(),
                           segment_lengths.begin(),
                           gather_locations.begin());
    }
    thrust::experimental::inclusive_segmented_scan(gather_locations.begin(), gather_locations.end(),
                                                   segments.begin(),
                                                   gather_locations.begin());
    
    // compute column entries and values of intermediate format
    cusp::array1d<IndexType,MemorySpace> I(coo_num_nonzeros);
    cusp::array1d<IndexType,MemorySpace> J(coo_num_nonzeros);
    cusp::array1d<ValueType,MemorySpace> V(coo_num_nonzeros);
    
    thrust::gather(I.begin(), I.end(),
                   segments.begin(),
                   A.row_indices.begin());

    thrust::gather(J.begin(), J.end(),
                   gather_locations.begin(),
                   B.column_indices.begin());
    {
        // TODO replace temp arrays with permutation_iterator
        cusp::array1d<ValueType,MemorySpace> temp1(coo_num_nonzeros);  // A_values[segments[n]]
        cusp::array1d<ValueType,MemorySpace> temp2(coo_num_nonzeros);  // B_values[gather_locations[n]]

        thrust::gather(temp1.begin(), temp1.end(), segments.begin(),         A.values.begin());
        thrust::gather(temp2.begin(), temp2.end(), gather_locations.begin(), B.values.begin());

        thrust::transform(temp1.begin(), temp1.end(),
                          temp2.begin(),
                          V.begin(),
                          thrust::multiplies<ValueType>());
    }

    // sort by (I,J)
    {
        // TODO use explicit permuation array
        thrust::sort_by_key(J.begin(), J.end(), thrust::make_zip_iterator(thrust::make_tuple(I.begin(), V.begin())));
        thrust::sort_by_key(I.begin(), I.end(), thrust::make_zip_iterator(thrust::make_tuple(J.begin(), V.begin())));
    }

    // compress duplicate (I,J) entries
    size_t NNZ = thrust::unique_by_key(thrust::make_zip_iterator(thrust::make_tuple(I.begin(), J.begin())),
                                       thrust::make_zip_iterator(thrust::make_tuple(I.end(), J.end())),
                                       V.begin(),
                                       thrust::equal_to< thrust::tuple<IndexType,IndexType> >(),
                                       thrust::plus<ValueType>()).second - V.begin();
    I.resize(NNZ);
    J.resize(NNZ);
    V.resize(NNZ);

    C.resize(A.num_rows, B.num_cols, NNZ);
    C.row_indices    = I;
    C.column_indices = J;
    C.values         = V;
}

template <typename ValueType, typename SpaceOrAlloc>
void matrix_multiply(const cusp::array2d<ValueType,SpaceOrAlloc>& A,
                     const cusp::array2d<ValueType,SpaceOrAlloc>& B,
                           cusp::array2d<ValueType,SpaceOrAlloc>& C)
{
    C.resize(A.num_rows, B.num_cols);

    for(size_t i = 0; i < C.num_rows; i++)
    {
        for(size_t j = 0; j < C.num_cols; j++)
        {
            ValueType v = 0;

            for(size_t k = 0; k < A.num_cols; k++)
                v += A(i,k) * B(k,j);
            
            C(i,j) = v;
        }
    }
}


template <typename SparseMatrixType, typename DenseMatrixType>
void _TestMatrixMultiply(SparseMatrixType test, DenseMatrixType A, DenseMatrixType B)
{
    DenseMatrixType C;
    matrix_multiply(A, B, C);

    SparseMatrixType _A(A), _B(B), _C;
    matrix_multiply(_A, _B, _C);

//    std::cout << "expected" << std::endl;
//    cusp::print_matrix(C);
//
//    std::cout << "result" << std::endl;
//    cusp::print_matrix(DenseMatrixType(_C));

    ASSERT_EQUAL(C == DenseMatrixType(_C), true);
}

template <class Space>
void TestMatrixMultiply(void)
{
    cusp::array2d<float,cusp::host_memory> A(3,2);
    A(0,0) = 1.0; A(0,1) = 2.0;
    A(1,0) = 3.0; A(1,1) = 0.0;
    A(2,0) = 5.0; A(2,1) = 6.0;
    
    cusp::array2d<float,cusp::host_memory> B(2,4);
    B(0,0) = 0.0; B(0,1) = 2.0; B(0,2) = 3.0; B(0,3) = 4.0;
    B(1,0) = 5.0; B(1,1) = 0.0; B(1,2) = 0.0; B(1,3) = 8.0;

    cusp::array2d<float,cusp::host_memory> C(2,2);
    C(0,0) = 0.0; C(0,1) = 0.0;
    C(1,0) = 3.0; C(1,1) = 5.0;
    
    cusp::array2d<float,cusp::host_memory> D(2,1);
    D(0,0) = 2.0;
    D(1,0) = 3.0;
    
    cusp::array2d<float,cusp::host_memory> E(2,2);
    E(0,0) = 0.0; E(0,1) = 0.0;
    E(1,0) = 0.0; E(1,1) = 0.0;
    
    cusp::array2d<float,cusp::host_memory> F(2,3);
    F(0,0) = 0.0; F(0,1) = 1.5; F(0,2) = 3.0;
    F(1,0) = 0.5; F(1,1) = 0.0; F(1,2) = 0.0;
   
    _TestMatrixMultiply(cusp::coo_matrix<int, float, Space>(), A, B);
    _TestMatrixMultiply(cusp::coo_matrix<int, float, Space>(), A, C);
    _TestMatrixMultiply(cusp::coo_matrix<int, float, Space>(), A, D);
    _TestMatrixMultiply(cusp::coo_matrix<int, float, Space>(), A, E);
    _TestMatrixMultiply(cusp::coo_matrix<int, float, Space>(), A, F);
    
    _TestMatrixMultiply(cusp::coo_matrix<int, float, Space>(), C, C);
    _TestMatrixMultiply(cusp::coo_matrix<int, float, Space>(), C, D);
    _TestMatrixMultiply(cusp::coo_matrix<int, float, Space>(), C, E);
    _TestMatrixMultiply(cusp::coo_matrix<int, float, Space>(), C, F);

    _TestMatrixMultiply(cusp::coo_matrix<int, float, Space>(), E, B);
    _TestMatrixMultiply(cusp::coo_matrix<int, float, Space>(), E, C);
    _TestMatrixMultiply(cusp::coo_matrix<int, float, Space>(), E, D);
    _TestMatrixMultiply(cusp::coo_matrix<int, float, Space>(), E, E);
    _TestMatrixMultiply(cusp::coo_matrix<int, float, Space>(), E, F);
    
    _TestMatrixMultiply(cusp::coo_matrix<int, float, Space>(), F, A);
}
DECLARE_HOST_DEVICE_UNITTEST(TestMatrixMultiply);

