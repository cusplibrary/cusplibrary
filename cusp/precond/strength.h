namespace cusp
{
namespace precond
{

/***
 *   Compute a strength of connection matrix using the standard symmetric measure.
 *   An off-diagonal connection A[i,j] is strong iff::
 *
 *   abs(A[i,j]) >= theta * sqrt( abs(A[i,i]) * abs(A[j,j]) )
 */
template <typename IndexType, typename ValueType>
void symmetric_strength_of_connection(	const cusp::coo_matrix<IndexType,ValueType,cusp::device_memory>& A, 
					cusp::coo_matrix<IndexType,ValueType,cusp::device_memory>& S,
					const ValueType theta = 0.0);

template <typename IndexType, typename ValueType>
void symmetric_strength_of_connection(	const cusp::coo_matrix<IndexType,ValueType,cusp::host_memory>& A, 
					cusp::coo_matrix<IndexType,ValueType,cusp::host_memory>& S,
					const ValueType theta = 0.0);

} // end namespace precond
} // end namespace cusp

#include <cusp/precond/detail/strength.inl>
