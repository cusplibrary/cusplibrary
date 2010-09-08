#include <cusp/csr_matrix.h>
#include <cusp/coo_matrix.h>
#include <thrust/count.h>

namespace cusp
{
namespace precond
{
	template <typename IndexType, typename ValueType, typename MemorySpace,
		  typename ArrayType>
	void mis_to_aggregates(const cusp::coo_matrix<IndexType,ValueType,MemorySpace>& C,
			       const ArrayType& mis,
				     ArrayType& aggregates);

	template <typename IndexType, typename ValueType,
		  typename ArrayType>
	void standard_aggregation(const cusp::coo_matrix<IndexType,ValueType,cusp::device_memory>& C,
					ArrayType& aggregates);

	template <typename IndexType, typename ValueType,
		  typename ArrayType>
	void standard_aggregation(const cusp::coo_matrix<IndexType,ValueType,cusp::host_memory>& C_coo,
					ArrayType& aggregates);

} // end namespace precond
} // end namespace cusp

#include <cusp/precond/detail/aggregate.inl>
