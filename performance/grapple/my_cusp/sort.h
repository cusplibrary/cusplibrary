#include <cusp/sort.h>

template <typename ArrayType>
void counting_sort(my_policy &exec,
                   ArrayType& v, typename ArrayType::value_type min, typename ArrayType::value_type max)
{
    std::cout << "calling my counting_sort\n";

    using cusp::system::detail::generic::counting_sort;

    counting_sort(exec.get(), v, min, max);
}

template <typename ArrayType1, typename ArrayType2>
void counting_sort_by_key(my_policy &exec,
                          ArrayType1& keys, ArrayType2& vals,
                          typename ArrayType1::value_type min, typename ArrayType1::value_type max)
{
    std::cout << "calling my counting_sort_by_key\n";

    using cusp::system::detail::generic::counting_sort_by_key;

    counting_sort_by_key(exec.get(), keys, vals, min, max);
}

template <typename ArrayType1, typename ArrayType2, typename ArrayType3>
void sort_by_row(my_policy &exec,
                 ArrayType1& row_indices, ArrayType2& column_indices, ArrayType3& values,
                 typename ArrayType1::value_type min_row,
                 typename ArrayType1::value_type max_row)
{
    std::cout << "calling my sort_by_row\n";

    using cusp::system::detail::generic::sort_by_row;

    sort_by_row(exec.get(),
                row_indices, column_indices, values,
                min_row, max_row);
}

template <typename ArrayType1, typename ArrayType2, typename ArrayType3>
void sort_by_row_and_column(my_policy &exec,
                            ArrayType1& row_indices, ArrayType2& column_indices, ArrayType3& values,
                            typename ArrayType1::value_type min_row,
                            typename ArrayType1::value_type max_row,
                            typename ArrayType2::value_type min_col,
                            typename ArrayType2::value_type max_col)
{
    std::cout << "calling my sort_by_row_and_column\n";

    using cusp::system::detail::generic::sort_by_row_and_column;

    sort_by_row_and_column(exec.get(),
                           row_indices, column_indices, values,
                           min_row, max_row, min_col, max_col);
}

