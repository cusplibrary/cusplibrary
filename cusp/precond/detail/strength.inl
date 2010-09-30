namespace cusp
{
namespace precond
{
namespace detail
{

template <typename ValueType>
struct absolute_value : public thrust::unary_function<ValueType,ValueType>
{
    ValueType operator()(const ValueType& x) const
    {
        return (x < 0) ? -x : x;
    }
};

template <typename IndexType, typename ValueType>
void symmetric_strength_of_connection(	const cusp::coo_matrix<IndexType,ValueType,cusp::host_memory>& A, 
					cusp::coo_matrix<IndexType,ValueType,cusp::host_memory>& S,
					const ValueType theta = 0.0 )
{
    //Sp,Sj form a CSR representation where the i-th row contains
    //the indices of all the strong connections from node i
    IndexType n_row = A.num_rows;
    IndexType n_col = A.num_cols;
    IndexType n_nnz = A.num_entries;

    //compute norm of diagonal values
    cusp::array1d<ValueType,cusp::host_memory> diagonal;
    cusp::detail::extract_diagonal(A, diagonal);
    cusp::array1d<ValueType,cusp::host_memory> diagonal_abs(A.num_rows,0);
    thrust::transform(diagonal.begin(), diagonal.end(), diagonal_abs.begin(), absolute_value<ValueType>());

    S.resize(n_row,n_col,n_nnz);

    cusp::array1d<IndexType,cusp::host_memory> row_offsets( A.num_rows+1, 0 );
    cusp::detail::indices_to_offsets(A.row_indices, row_offsets);

    IndexType nnz = 0;
    for(IndexType i = 0; i < n_row; i++){
        ValueType eps_Aii = theta*theta*diagonal_abs[i];

        for(IndexType jj = row_offsets[i]; jj < row_offsets[i+1]; jj++){
            const IndexType   j = A.column_indices[jj];
            const ValueType Aij = A.values[jj];

            if(i == j){continue;}  //skip diagonal

            //  |A(i,j)| >= theta * sqrt(|A(i,i)|*|A(j,j)|)
            if(Aij*Aij >= eps_Aii * diagonal_abs[j]){
                S.row_indices[nnz] =   i;
                S.column_indices[nnz] =   j;
                S.values[nnz] = Aij;
                nnz++;
            }
        }
    }

    S.resize(n_row,n_col,nnz);
}

template <typename IndexType, typename ValueType>
void symmetric_strength_of_connection(	const cusp::coo_matrix<IndexType,ValueType,cusp::device_memory>& A, 
					cusp::coo_matrix<IndexType,ValueType,cusp::device_memory>& S,
					const ValueType theta = 0.0 )
{
	// TODO implement device method to avoid transfers
	const cusp::coo_matrix<IndexType,ValueType,cusp::host_memory> A_host(A);
	cusp::coo_matrix<IndexType,ValueType,cusp::host_memory> S_host;
	symmetric_strength_of_connection( A_host, S_host );
	S = S_host;
}


} // end namepace detail
} // end namespace precond
} // end namespace cusp
