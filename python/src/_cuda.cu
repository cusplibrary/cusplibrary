#include <cstddef>
#include <cstdint>

#include <thrust/device_ptr.h>

#include <cusp/array1d.h>
#include <cusp/csr_matrix.h>
#include <cusp/multiply.h>

namespace {

template <typename ValueType>
void spmv_cuda_impl(std::size_t nrow, std::size_t ncol, std::size_t nnz,
                    const std::int32_t* row_offsets,
                    const std::int32_t* column_indices,
                    const ValueType* values, const ValueType* x,
                    ValueType* y) {
  using IndexPtr = thrust::device_ptr<std::int32_t>;
  using ValuePtr = thrust::device_ptr<ValueType>;
  using IndexView = cusp::array1d_view<IndexPtr>;
  using ValueView = cusp::array1d_view<ValuePtr>;

  IndexPtr ro_p(const_cast<std::int32_t*>(row_offsets));
  IndexPtr ci_p(const_cast<std::int32_t*>(column_indices));
  ValuePtr vs_p(const_cast<ValueType*>(values));
  ValuePtr x_p(const_cast<ValueType*>(x));
  ValuePtr y_p(y);

  IndexView ro_view(ro_p, ro_p + nrow + 1);
  IndexView ci_view(ci_p, ci_p + nnz);
  ValueView vs_view(vs_p, vs_p + nnz);
  ValueView x_view(x_p, x_p + ncol);
  ValueView y_view(y_p, y_p + nrow);

  cusp::csr_matrix_view<IndexView, IndexView, ValueView> A(
      nrow, ncol, nnz, ro_view, ci_view, vs_view);

  cusp::multiply(A, x_view, y_view);
}

}  // namespace

extern "C" {

void pycusp_spmv_cuda_f32(std::size_t nrow, std::size_t ncol, std::size_t nnz,
                           const std::int32_t* row_offsets,
                           const std::int32_t* column_indices,
                           const float* values, const float* x, float* y) {
  spmv_cuda_impl<float>(nrow, ncol, nnz, row_offsets, column_indices, values,
                        x, y);
}

void pycusp_spmv_cuda_f64(std::size_t nrow, std::size_t ncol, std::size_t nnz,
                           const std::int32_t* row_offsets,
                           const std::int32_t* column_indices,
                           const double* values, const double* x, double* y) {
  spmv_cuda_impl<double>(nrow, ncol, nnz, row_offsets, column_indices, values,
                         x, y);
}

}  // extern "C"
