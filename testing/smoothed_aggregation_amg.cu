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

#include <thrust/extrema.h>
#include <thrust/transform.h>
#include <thrust/gather.h>

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

    thrust::next::gather(idx2.begin(), idx2.end(),
                         mis_enum.begin(),
                         aggregates.begin());
}

template <typename IndexType, typename ValueType, typename MemorySpace,
          typename ArrayType>
void standard_aggregation(const cusp::coo_matrix<IndexType,ValueType,MemorySpace>& C,
                                ArrayType& aggregates)
{
    // TODO check sizes

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

void TestStandardAggregation(void)
{
    cusp::coo_matrix<int,float,cusp::device_memory> A;
    cusp::gallery::poisson5pt(A, 10, 10);

    cusp::array1d<int,cusp::device_memory> aggregates(A.num_rows);
    standard_aggregation(A, aggregates);

//    cusp::print_matrix(aggregates);
}
DECLARE_UNITTEST(TestStandardAggregation);

