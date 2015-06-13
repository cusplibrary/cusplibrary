/*
 *  Copyright 2008-2014 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

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

template<typename T>
struct distance_filter_functor
{
  __host__ __device__
  T operator()(const T& A_val, const T& S_val) const
  {
      return A_val >= S_val ? 0 : A_val;
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
void apply_distance_filter(const double epsilon, MatrixType A)
{
    using namespace thrust::placeholders;

    typedef typename MatrixType::index_type IndexType;
    typedef typename MatrixType::value_type ValueType;
    typedef typename MatrixType::memory_space MemorySpace;

    const size_t N = A.num_rows;

    cusp::array1d<ValueType, MemorySpace> smallest_per_row(N, ValueType(0));

    thrust::reduce_by_key(A.row_indices.begin(), A.row_indices.end(), A.values.begin(),
                          thrust::make_discard_iterator(), smallest_per_row.begin(),
                          thrust::equal_to<IndexType>(), thrust::minimum<ValueType>());

    thrust::transform(smallest_per_row.begin(), smallest_per_row.end(),
                      thrust::constant_iterator<ValueType>(epsilon),
                      smallest_per_row.begin(),
                      thrust::multiplies<ValueType>());

    thrust::transform(A.values.begin(), A.values.end(),
                      thrust::make_permutation_iterator(smallest_per_row.begin(), A.row_indices.begin()),
                      A.values.begin(),
                      distance_filter_functor<ValueType>());
}

template<typename DerivedPolicy, typename MatrixType1, typename MatrixType2, typename ArrayType>
typename thrust::detail::enable_if_convertible<typename ArrayType::format,cusp::array1d_format>::type
evolution_strength_of_connection(thrust::execution_policy<DerivedPolicy> &exec,
                                 const MatrixType1& A, MatrixType2& S, const ArrayType& B,
                                 double rho_DinvA, const double epsilon, cusp::coo_format)
{
    using namespace thrust::placeholders;

    typedef typename MatrixType1::index_type IndexType;
    typedef typename MatrixType1::value_type ValueType;
    typedef typename MatrixType1::memory_space MemorySpace;
    typedef typename MatrixType1::const_view MatrixViewType;

    const size_t N = A.num_rows;
    const size_t M = A.num_entries;

    cusp::array1d<ValueType, MemorySpace> D(N);
    cusp::array1d<ValueType, MemorySpace> DAtilde(N);
    // cusp::array1d<ValueType, MemorySpace> largest_per_row(N, 0);

    cusp::array1d<ValueType, MemorySpace> data(M);
    cusp::array1d<ValueType, MemorySpace> angle(M);
    cusp::array1d<ValueType, MemorySpace> Atilde_symmetric(M);
    cusp::array1d<ValueType, MemorySpace> Atilde_values(M);

    cusp::array1d<bool, MemorySpace> filter(M);

    cusp::array1d<ValueType, MemorySpace> Bmat_forscaling(B);
    cusp::array1d<ValueType, MemorySpace> Dinv_A_values(A.values);

    cusp::array1d<IndexType, MemorySpace> permutation(M);
    cusp::array1d<IndexType, MemorySpace> indices(M);
    cusp::array1d<IndexType, MemorySpace> A_row_offsets(N + 1);

    MatrixViewType Dinv_A(N, N, M, A.row_indices, A.column_indices, Dinv_A_values);
    MatrixViewType Atilde(N, N, M, A.row_indices, A.column_indices, Atilde_values);

    cusp::extract_diagonal(exec, A, D);

    // scale the rows of D_inv_S by D^-1
    thrust::transform(exec,
                      Dinv_A_values.begin(), Dinv_A_values.end(),
                      thrust::make_permutation_iterator(D.begin(), A.row_indices.begin()),
                      Dinv_A_values.begin(),
                      thrust::divides<ValueType>());

    if(rho_DinvA == 0.0)
    {
      rho_DinvA = cusp::eigen::ritz_spectral_radius(Dinv_A, 8);
      std::cout << "rho(D^-1A) = " << rho_DinvA << std::endl;
    }

    cusp::indices_to_offsets(exec, A.row_indices, A_row_offsets);

    thrust::transform(exec,
                      thrust::make_zip_iterator(thrust::make_tuple(A.row_indices.begin(), A.column_indices.begin(), Dinv_A_values.begin())),
                      thrust::make_zip_iterator(thrust::make_tuple(A.row_indices.begin(), A.column_indices.begin(), Dinv_A_values.begin())) + M,
                      Dinv_A_values.begin(), Atilde_functor<ValueType>(rho_DinvA));

    incomplete_inner_functor<ValueType> incomp_op(thrust::raw_pointer_cast(&A_row_offsets[0]),
                                                  thrust::raw_pointer_cast(&A.column_indices[0]),
                                                  thrust::raw_pointer_cast(&Dinv_A_values[0]));

    thrust::transform(exec,
                      thrust::make_zip_iterator(thrust::make_tuple(A.row_indices.begin(), A.column_indices.begin())),
                      thrust::make_zip_iterator(thrust::make_tuple(A.row_indices.begin(), A.column_indices.begin())) + A.num_entries,
                      Atilde_values.begin(),
                      incomp_op);

    thrust::replace(exec, Bmat_forscaling.begin(), Bmat_forscaling.end(), 0, 1);

    cusp::extract_diagonal(exec, Atilde, DAtilde);

    cusp::copy(exec, Atilde_values, data);

    thrust::transform(exec,
                      thrust::constant_iterator<ValueType>(1), thrust::constant_iterator<ValueType>(1) + M,
                      thrust::make_permutation_iterator(DAtilde.begin(), A.row_indices.begin()),
                      Atilde_values.begin(),
                      thrust::multiplies<ValueType>());

    thrust::transform(exec,
                      Atilde_values.begin(), Atilde_values.end(),
                      thrust::make_permutation_iterator(Bmat_forscaling.begin(), A.column_indices.begin()),
                      Atilde_values.begin(),
                      thrust::multiplies<ValueType>());

    // Calculate approximation ratio
    cusp::blas::xmy(exec, data, Atilde_values, angle);
    thrust::transform(exec, angle.begin(), angle.end(), thrust::constant_iterator<ValueType>(0), filter.begin(), thrust::less<ValueType>());

    thrust::transform(exec, Atilde_values.begin(), Atilde_values.end(), data.begin(), Atilde_values.begin(), thrust::divides<ValueType>());

    // Set small ratios and large angles to weak
    thrust::transform(exec,
                      thrust::make_zip_iterator(thrust::make_tuple(angle.begin(), weak_ratios.begin())),
                      thrust::make_zip_iterator(thrust::make_tuple(angle.begin(), weak_ratios.begin())) + M,
                      Atilde_values.begin(),
                      filter_small_ratios_and_large_angles<ValueType>());

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

    // Apply distance filter
    if(epsilon != std::numeric_limits<ValueType>::infinity())
    {
        apply_distance_filter(epsilon,
                              cusp::make_coo_matrix_view(N, N, M, A.row_indices, A.column_indices, Atilde_symmetric));
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

    // // Standardized strength values require small values be weak and large values be strong (invert).
    // thrust::transform(exec, Atilde_symmetric.begin(), Atilde_symmetric.end(), Atilde_symmetric.begin(), conditional_invert<ValueType>());
    //
    // thrust::reduce_by_key(exec,
    //                       A.row_indices.begin(), A.row_indices.end(), Atilde_symmetric.begin(),
    //                       thrust::make_discard_iterator(), largest_per_row.begin(),
    //                       thrust::equal_to<IndexType>(), thrust::maximum<ValueType>());
    //
    // thrust::transform(exec,
    //                   Atilde_symmetric.begin(), Atilde_symmetric.end(),
    //                   thrust::make_permutation_iterator(largest_per_row.begin(), A.row_indices.begin()),
    //                   Atilde_symmetric.begin(),
    //                   thrust::divides<ValueType>());

    size_t num_zeros = thrust::count(Atilde_symmetric.begin(), Atilde_symmetric.end(), ValueType(0));
    std::cout << "Filtered " << num_zeros << " edges" << std::endl;

    S.resize(N, N, M - num_zeros);
    thrust::copy_if(exec, A.row_indices.begin(), A.row_indices.end(), Atilde_symmetric.begin(), S.row_indices.begin(), _1 != 0);
    thrust::copy_if(exec, A.column_indices.begin(), A.column_indices.end(), Atilde_symmetric.begin(), S.column_indices.begin(), _1 != 0);
}

template<typename DerivedPolicy, typename MatrixType1, typename MatrixType2, typename ArrayType>
typename thrust::detail::enable_if_convertible<typename ArrayType::format,cusp::array1d_format>::type
evolution_strength_of_connection(thrust::execution_policy<DerivedPolicy> &exec,
                                 const MatrixType1& A, MatrixType2& S, const ArrayType& B,
                                 double rho_DinvA, const double epsilon, cusp::csr_format)
{
    typedef typename MatrixType1::index_type IndexType;
    typedef typename MatrixType1::value_type ValueType;
    typedef typename MatrixType1::memory_space MemorySpace;
    typedef typename MatrixType1::const_view MatrixViewType;

    cusp::array1d<IndexType, MemorySpace> A_row_indices(A.num_entries + 1);
    cusp::offsets_to_indices(A.row_offsets, A_row_indices);

    cusp::coo_matrix<IndexType, ValueType, MemorySpace> S_;

    evolution_strength_of_connection(exec,
                                     cusp::make_coo_matrix_view(A.num_rows, A.num_cols, A.num_entries,
                                                                A_row_indices, A.column_indices, A.values),
                                     S_, B, rho_DinvA, epsilon, cusp::coo_format());

    S = S_;
}

template<typename DerivedPolicy, typename MatrixType1, typename MatrixType2, typename ArrayType>
typename thrust::detail::enable_if_convertible<typename ArrayType::format,cusp::array1d_format>::type
evolution_strength_of_connection(thrust::execution_policy<DerivedPolicy> &exec,
                                 const MatrixType1& A, MatrixType2& S, const ArrayType& B,
                                 double rho_DinvA, const double epsilon)
{
    typedef typename MatrixType1::format Format;

    Format format;

    evolution_strength_of_connection(exec, A, S, B, rho_DinvA, epsilon, format);
}

} // end namespace detail
} // end namespace aggregation
} // end namespace precond
} // end namespace cusp

