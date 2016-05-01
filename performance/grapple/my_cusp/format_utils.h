#include <cusp/format_utils.h>

template <typename Matrix, typename Array>
void extract_diagonal(my_policy &exec,
                      const Matrix& A, Array& output)
{
    std::cout << "calling my extract_diagonal\n";

    using cusp::system::detail::generic::extract_diagonal;

    typedef typename Matrix::format Format;

    Format format;

    output.resize(thrust::min(A.num_rows, A.num_cols));

    // dispatch on matrix format
    extract_diagonal(exec.get(),
                     A, output, format);
}

template <typename OffsetArray, typename IndexArray>
void offsets_to_indices(my_policy &exec,
                        const OffsetArray& offsets, IndexArray& indices)
{
    std::cout << "calling my offsets_to_indices\n";

    using cusp::system::detail::generic::offsets_to_indices;

    offsets_to_indices(exec.get(),
                       offsets, indices);
}

template <typename IndexArray, typename OffsetArray>
void indices_to_offsets(my_policy &exec,
                        const IndexArray& indices, OffsetArray& offsets)
{
    std::cout << "calling my indices_to_offsets\n";

    using cusp::system::detail::generic::indices_to_offsets;

    indices_to_offsets(exec.get(),
                       indices, offsets);
}

template <typename ArrayType1, typename ArrayType2>
size_t count_diagonals(my_policy &exec,
                       const size_t num_rows,
                       const size_t num_cols,
                       const ArrayType1& row_indices,
                       const ArrayType2& column_indices)
{
    std::cout << "calling my count_diagonals\n";

    using cusp::system::detail::generic::count_diagonals;

    return count_diagonals(exec.get(),
                           num_rows, num_cols, row_indices, column_indices);
}

template <typename ArrayType>
size_t compute_max_entries_per_row(my_policy &exec,
                                   const ArrayType& row_offsets)
{
    std::cout << "calling my compute_max_entries_per_row\n";

    using cusp::system::detail::generic::compute_max_entries_per_row;

    return compute_max_entries_per_row(exec.get(),
                                       row_offsets);
}

template <typename ArrayType>
size_t compute_optimal_entries_per_row(my_policy &exec,
                                       const ArrayType& row_offsets,
                                       float relative_speed,
                                       size_t breakeven_threshold)
{
    std::cout << "calling my compute_optimal_entries_per_row\n";

    using cusp::system::detail::generic::compute_optimal_entries_per_row;

    return compute_optimal_entries_per_row(exec.get(),
                                           row_offsets, relative_speed, breakeven_threshold);
}

