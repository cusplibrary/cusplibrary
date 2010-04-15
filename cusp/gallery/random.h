#include <cusp/coo_matrix.h>
#include <thrust/unique.h>
#include <thrust/sort.h>

#include <stdlib.h> // XXX remove when we switch RNGs

namespace cusp
{
namespace gallery
{

// TODO use thrust RNGs, add seed parameter defaulting to num_rows ^ num_cols ^ num_samples
template <class MatrixType>
void random(size_t num_rows, size_t num_cols, size_t num_samples, MatrixType& output)
{
    typedef typename MatrixType::index_type IndexType;
    typedef typename MatrixType::value_type ValueType;

    cusp::coo_matrix<IndexType,ValueType,cusp::host_memory> coo(num_rows, num_cols, num_samples);

    srand(num_rows ^ num_cols ^ num_samples);

    for(size_t n = 0; n < num_samples; n++)
    {
        coo.row_indices[n]    = rand() % num_rows;
        coo.column_indices[n] = rand() % num_cols;
        coo.values[n]         = ValueType(1);
    }

    // sort indices by (row,column)
    thrust::sort(thrust::make_zip_iterator(thrust::make_tuple(coo.row_indices.begin(), coo.column_indices.begin())),
                 thrust::make_zip_iterator(thrust::make_tuple(coo.row_indices.end(),   coo.column_indices.end())));

    size_t num_entries = thrust::unique(thrust::make_zip_iterator(thrust::make_tuple(coo.row_indices.begin(), coo.column_indices.begin())),
                                        thrust::make_zip_iterator(thrust::make_tuple(coo.row_indices.end(),   coo.column_indices.end())))
                         - thrust::make_zip_iterator(thrust::make_tuple(coo.row_indices.begin(), coo.column_indices.begin()));

    coo.resize(num_rows, num_cols, num_entries);
    
    output = coo;
}

} // end namespace gallery
} // end namespace cusp

