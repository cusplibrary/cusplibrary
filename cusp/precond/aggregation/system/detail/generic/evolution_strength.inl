#include <cusp/csr_matrix.h>
#include <cusp/format_utils.h>
#include <cusp/linear_operator.h>
#include <cusp/multiply.h>

#include <cusp/eigen/spectral_radius.h>

#include <thrust/functional.h>
#include <thrust/iterator/permutation_iterator.h>

namespace cusp
{
namespace precond
{
namespace aggregation
{
namespace detail
{

template<typename ValueType>
struct Atilde_functor
{
    const ValueType rho_DinvA;

    Atilde_functor(const ValueType rho_DinvA)
        : rho_DinvA(rho_DinvA)
    {}

    template <typename Tuple>
    __host__ __device__
    ValueType operator()(const Tuple& t) const
    {
        int row = thrust::get<0>(t);
        int col = thrust::get<1>(t);
        ValueType val = thrust::get<2>(t);
        ValueType temp = row == col;

        return temp - (1.0/rho_DinvA)*val;
    }
};

template<typename ValueType>
struct approx_error : public thrust::unary_function<ValueType,ValueType>
{
    __host__ __device__
    ValueType operator()(const ValueType scale) const
    {
        return abs(1.0 - scale);
    }
};

template<typename ValueType>
struct conditional_invert : public thrust::unary_function<ValueType,ValueType>
{
    __host__ __device__
    ValueType operator()(const ValueType val) const
    {
        return val != 0.0 ? 1.0/val : val;
    }
};

template<typename ValueType>
struct filter_small_ratios_and_large_angles : public thrust::unary_function<ValueType,ValueType>
{
    template<typename Tuple>
    __host__ __device__
    ValueType operator()(const Tuple& t) const
    {
        ValueType angle = thrust::get<0>(t);
        ValueType ratio = thrust::get<1>(t);

        return ((angle < 0.0) || (abs(ratio) < 1e-4)) ? 0 : ratio;
    }
};

template<typename ValueType>
struct set_perfect : public thrust::unary_function<ValueType,ValueType>
{
    const ValueType eps;

    set_perfect(void) : eps(std::sqrt(std::numeric_limits<ValueType>::epsilon())) {}

    __host__ __device__
    ValueType operator()(const ValueType val) const
    {
        return ((val < eps) && (val != 0)) ? 1e-4 : val;
    }
};

template<typename ValueType>
struct incomplete_inner_functor
{
    const int *Ap, *Aj;
    const ValueType *Ax;

    incomplete_inner_functor(const int *Ap, const int *Aj, const ValueType *Ax)
        : Ap(Ap), Aj(Aj), Ax(Ax)
    {}

    template <typename Tuple>
    __host__ __device__
    ValueType operator()(const Tuple& t) const
    {
        ValueType sum = 0.0;

        int row   = thrust::get<0>(t);
        int col   = thrust::get<1>(t);

        int A_pos = Ap[row];
        int A_end = Ap[row+1];
        int B_pos = Ap[col];
        int B_end = Ap[col+1];

        //while not finished with either A[row,:] or B[:,col]
        while(A_pos < A_end && B_pos < B_end) {
            int A_j = Aj[A_pos];
            int B_j = Aj[B_pos];

            if(A_j == B_j) {
                sum += Ax[A_pos] * Ax[B_pos];
                A_pos++;
                B_pos++;
            } else if (A_j < B_j) {
                A_pos++;
            } else {
                //B_j < A_j
                B_pos++;
            }
        }

        return sum;
    }
};

template<typename MatrixType>
void apply_distance_filter(MatrixType& A)
{
}

template<typename DerivedPolicy, typename MatrixType1, typename MatrixType2, typename ArrayType>
void evolution_strength_of_connection(thrust::execution_policy<DerivedPolicy> &exec,
                                      const MatrixType1& A, MatrixType2& S, const ArrayType& B,
                                      const double rho_DinvA, const double epsilon)
{
    using namespace thrust::placeholders;

    typedef typename MatrixType1::index_type IndexType;
    typedef typename MatrixType1::value_type ValueType;
    typedef typename MatrixType1::memory_space MemorySpace;
    typedef typename MatrixType1::const_view MatrixViewType;

    const size_t N = A.num_rows;
    const size_t M = A.num_entries;

    cusp::array1d<ValueType, MemorySpace> D(N);
    cusp::array1d<ValueType, MemorySpace> Dinv_A_values(A.values);
    cusp::array1d<ValueType, MemorySpace> Atilde_values(M, 0);
    cusp::array1d<ValueType, MemorySpace> Bmat_forscaling(B);
    cusp::array1d<ValueType, MemorySpace> DAtilde(N);
    cusp::array1d<ValueType, MemorySpace> data(M);
    cusp::array1d<ValueType, MemorySpace> angle(M);
    cusp::array1d<ValueType, MemorySpace> largest_per_row(N, 0);
    cusp::array1d<ValueType, MemorySpace> Atilde_symmetric(M);

    cusp::array1d<IndexType, MemorySpace> permutation(M);
    cusp::array1d<IndexType, MemorySpace> indices(M);
    cusp::array1d<IndexType, MemorySpace> A_row_offsets(N + 1);

    MatrixViewType Dinv_A(A.num_rows, A.num_cols, A.num_entries, A_row_offsets, A.column_indices, Dinv_A_values);
    MatrixViewType Atilde(A.num_rows, A.num_cols, A.num_entries, A_row_offsets, A.column_indices, Atilde_values);

    cusp::extract_diagonal(exec, A, D);

    cusp::indices_to_offsets(exec, A.row_indices, A_row_offsets);

    // scale the rows of D_inv_S by D^-1
    thrust::transform(exec,
                      Dinv_A_values.begin(), Dinv_A_values.end(),
                      thrust::make_permutation_iterator(D.begin(), A.row_indices.begin()),
                      Dinv_A_values.begin(),
                      thrust::divides<ValueType>());

    thrust::transform(exec,
                      thrust::make_zip_iterator(thrust::make_tuple(A.row_indices.begin(), A.column_indices.begin(), Dinv_A_values.begin())),
                      thrust::make_zip_iterator(thrust::make_tuple(A.row_indices.begin(), A.column_indices.begin(), Dinv_A_values.begin())) + A.num_entries,
                      Dinv_A_values.begin(), Atilde_functor<ValueType>(rho_DinvA));

    thrust::transform(exec,
                      thrust::make_zip_iterator(thrust::make_tuple(A.row_indices.begin(), A.column_indices.begin())),
                      thrust::make_zip_iterator(thrust::make_tuple(A.row_indices.begin(), A.column_indices.begin())) + A.num_entries,
                      Atilde_values.begin(),
                      incomplete_inner_functor<ValueType>(thrust::raw_pointer_cast(&A_row_offsets[0]), thrust::raw_pointer_cast(&A.column_indices[0]), thrust::raw_pointer_cast(&Dinv_A_values[0])));

    thrust::replace(exec, Bmat_forscaling.begin(), Bmat_forscaling.end(), 0, 1);

    cusp::extract_diagonal(exec, Atilde, DAtilde);

    cusp::copy(exec, Atilde_values, data);
    cusp::blas::fill(exec, Atilde_values, ValueType(1));

    thrust::transform(exec,
                      Atilde_values.begin(), Atilde_values.end(),
                      thrust::make_permutation_iterator(DAtilde.begin(), A.row_indices.begin()),
                      Atilde_values.begin(),
                      thrust::multiplies<ValueType>());
    thrust::transform(exec,
                      Atilde_values.begin(), Atilde_values.end(),
                      thrust::make_permutation_iterator(Bmat_forscaling.begin(), A.column_indices.begin()),
                      Atilde_values.begin(),
                      thrust::multiplies<ValueType>());

    cusp::blas::xmy(data, Atilde_values, angle);

    // Calculate approximation ratio
    thrust::transform(exec, Atilde_values.begin(), Atilde_values.end(), data.begin(), Atilde_values.begin(), thrust::divides<ValueType>());
    // Set small ratios and large angles to weak
    thrust::transform(exec,
                      thrust::make_zip_iterator(thrust::make_tuple(angle.begin(), Atilde_values.begin())),
                      thrust::make_zip_iterator(thrust::make_tuple(angle.begin(), Atilde_values.begin())) + A.num_entries,
                      Atilde_values.begin(), filter_small_ratios_and_large_angles<ValueType>());
    // Calculate approximation error
    thrust::transform(exec, Atilde_values.begin(), Atilde_values.end(), Atilde_values.begin(), approx_error<ValueType>());

    // Set near perfect connections to 1e-4
    thrust::transform(exec, Atilde_values.begin(), Atilde_values.end(), Atilde_values.begin(), set_perfect<ValueType>());

    // symmetrize measure
    thrust::sequence(exec, permutation.begin(), permutation.end());
    cusp::copy(exec, A.column_indices, indices);
    thrust::sort_by_key(exec, indices.begin(), indices.end(), permutation.begin());

    thrust::transform(exec,
                      Atilde_values.begin(), Atilde_values.end(),
                      thrust::make_permutation_iterator(Atilde_values.begin(), permutation.begin()),
                      Atilde_symmetric.begin(),
                      0.5 * (_1 + _2));

    if(epsilon != std::numeric_limits<ValueType>::infinity())
    {
        MatrixViewType Atilde_symmetric_matrix(A.num_rows, A.num_cols, A.num_entries, A_row_offsets, A.column_indices, Atilde_symmetric);
        apply_distance_filter(Atilde_symmetric_matrix);
    }

    // Set diagonal to 1.0, as each point is strongly connected to itself
    thrust::transform_if(exec,
                         Atilde_symmetric.begin(), Atilde_symmetric.end(),
                         thrust::make_transform_iterator(
                             thrust::make_zip_iterator(
                                 thrust::make_tuple(A.row_indices.begin(), A.column_indices.begin())),
                             cusp::detail::equal_tuple_functor<IndexType>()),
                         Atilde_symmetric.begin(),
                         _1 = 1, thrust::identity<bool>());

    // Standardized strength values require small values be weak and large values be strong (invert).
    thrust::transform(exec, Atilde_symmetric.begin(), Atilde_symmetric.end(), Atilde_symmetric.begin(), conditional_invert<ValueType>());

    thrust::reduce_by_key(exec,
                          A.row_indices.begin(), A.row_indices.end(), Atilde_symmetric.begin(),
                          thrust::make_discard_iterator(), largest_per_row.begin(),
                          thrust::equal_to<IndexType>(), thrust::maximum<ValueType>());

    thrust::transform(exec,
                      Atilde_symmetric.begin(), Atilde_symmetric.end(),
                      thrust::make_permutation_iterator(largest_per_row.begin(), A.row_indices.begin()),
                      Atilde_symmetric.begin(), thrust::divides<ValueType>());

    cusp::copy(exec, A, S);
    cusp::copy(exec, Atilde_symmetric, S.values);
}

} // end namespace detail
} // end namespace aggregation
} // end namespace precond
} // end namespace cusp

