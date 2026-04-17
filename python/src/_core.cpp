#include <cstdint>
#include <stdexcept>
#include <string>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cusp/array1d.h>
#include <cusp/csr_matrix.h>
#include <cusp/multiply.h>

namespace nb = nanobind;

namespace {

using I32 = std::int32_t;
using NBArrayI32 = nb::ndarray<I32, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using NBArrayAnyRO = nb::ndarray<nb::ro, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using NBArrayAnyRW = nb::ndarray<nb::ndim<1>, nb::c_contig, nb::device::cpu>;

enum class ValueDType { Float32, Float64 };

template <typename Array>
ValueDType classify_value_dtype(const Array& arr, const char* name) {
  const auto dt = arr.dtype();
  if (dt == nb::dtype<float>()) return ValueDType::Float32;
  if (dt == nb::dtype<double>()) return ValueDType::Float64;
  throw nb::type_error((std::string(name) + " must be float32 or float64").c_str());
}

class CsrMatrix {
 public:
  CsrMatrix(std::size_t num_rows, std::size_t num_cols,
            NBArrayI32 row_offsets, NBArrayI32 column_indices,
            NBArrayAnyRO values)
      : num_rows_(num_rows),
        num_cols_(num_cols),
        row_offsets_(std::move(row_offsets)),
        column_indices_(std::move(column_indices)),
        values_(std::move(values)) {
    if (row_offsets_.shape(0) != num_rows_ + 1) {
      throw std::invalid_argument("row_offsets length must equal num_rows + 1");
    }
    const std::size_t nnz = column_indices_.shape(0);
    if (values_.shape(0) != nnz) {
      throw std::invalid_argument("column_indices and values must have equal length");
    }
    if (num_rows_ > 0) {
      const I32 last = row_offsets_(num_rows_);
      if (last < 0 || static_cast<std::size_t>(last) != nnz) {
        throw std::invalid_argument(
            "row_offsets[-1] must equal len(values) (the number of nonzeros)");
      }
      if (row_offsets_(0) != 0) {
        throw std::invalid_argument("row_offsets[0] must be 0");
      }
    }
    num_entries_ = nnz;
    dtype_ = classify_value_dtype(values_, "values");
  }

  std::size_t num_rows() const { return num_rows_; }
  std::size_t num_cols() const { return num_cols_; }
  std::size_t num_entries() const { return num_entries_; }
  ValueDType dtype() const { return dtype_; }

  const I32* row_offsets_ptr() const { return row_offsets_.data(); }
  const I32* column_indices_ptr() const { return column_indices_.data(); }
  const void* values_ptr() const { return values_.data(); }

 private:
  std::size_t num_rows_;
  std::size_t num_cols_;
  std::size_t num_entries_;
  ValueDType dtype_;
  NBArrayI32 row_offsets_;
  NBArrayI32 column_indices_;
  NBArrayAnyRO values_;
};

template <typename ValueType>
void spmv_dispatch(const CsrMatrix& A, const ValueType* xp, ValueType* yp) {
  // CUSP expects non-const pointers on the view types, so cast away const.
  // The underlying storage is owned by NumPy; x is read-only.
  using IndexPtr = I32*;
  using ValuePtr = ValueType*;
  using IndexView = cusp::array1d_view<IndexPtr>;
  using ValueView = cusp::array1d_view<ValuePtr>;

  const std::size_t nrow = A.num_rows();
  const std::size_t ncol = A.num_cols();
  const std::size_t nnz = A.num_entries();

  IndexView row_offsets(const_cast<IndexPtr>(A.row_offsets_ptr()),
                        const_cast<IndexPtr>(A.row_offsets_ptr()) + nrow + 1);
  IndexView column_indices(
      const_cast<IndexPtr>(A.column_indices_ptr()),
      const_cast<IndexPtr>(A.column_indices_ptr()) + nnz);
  ValueView values(const_cast<ValuePtr>(static_cast<const ValueType*>(A.values_ptr())),
                   const_cast<ValuePtr>(static_cast<const ValueType*>(A.values_ptr())) + nnz);

  cusp::csr_matrix_view<IndexView, IndexView, ValueView> Aview(
      nrow, ncol, nnz, row_offsets, column_indices, values);

  ValueView x_view(const_cast<ValuePtr>(xp), const_cast<ValuePtr>(xp) + ncol);
  ValueView y_view(yp, yp + nrow);

  cusp::multiply(Aview, x_view, y_view);
}

void spmv(const CsrMatrix& A, NBArrayAnyRO x, NBArrayAnyRW y) {
  if (x.shape(0) != A.num_cols()) {
    throw std::invalid_argument("x length must equal A.num_cols");
  }
  if (y.shape(0) != A.num_rows()) {
    throw std::invalid_argument("y length must equal A.num_rows");
  }

  const ValueDType x_dt = classify_value_dtype(x, "x");
  const ValueDType y_dt = classify_value_dtype(y, "y");
  if (x_dt != A.dtype() || y_dt != A.dtype()) {
    throw nb::type_error(
        "dtype mismatch: A, x, y must all be the same float32 or float64 dtype");
  }

  nb::gil_scoped_release release;
  switch (A.dtype()) {
    case ValueDType::Float32:
      spmv_dispatch<float>(A,
                           static_cast<const float*>(x.data()),
                           static_cast<float*>(y.data()));
      break;
    case ValueDType::Float64:
      spmv_dispatch<double>(A,
                            static_cast<const double*>(x.data()),
                            static_cast<double*>(y.data()));
      break;
  }
}

}  // namespace

NB_MODULE(_core, m) {
  m.doc() = "pycusp._core: nanobind bindings for CUSP sparse algorithms";
  m.attr("__version__") = "0.1.0";

  nb::class_<CsrMatrix>(m, "CsrMatrix",
                        "Non-owning CSR view over NumPy arrays.")
      .def(nb::init<std::size_t, std::size_t, NBArrayI32, NBArrayI32,
                    NBArrayAnyRO>(),
           nb::arg("num_rows"), nb::arg("num_cols"), nb::arg("row_offsets"),
           nb::arg("column_indices"), nb::arg("values"))
      .def_prop_ro("num_rows", &CsrMatrix::num_rows)
      .def_prop_ro("num_cols", &CsrMatrix::num_cols)
      .def_prop_ro("num_entries", &CsrMatrix::num_entries)
      .def_prop_ro("shape", [](const CsrMatrix& A) {
        return nb::make_tuple(A.num_rows(), A.num_cols());
      })
      .def_prop_ro("dtype", [](const CsrMatrix& A) {
        return A.dtype() == ValueDType::Float32 ? "float32" : "float64";
      });

  m.def("spmv", &spmv, nb::arg("A"), nb::arg("x"), nb::arg("y"),
        "Compute y = A * x for a CSR matrix A. y is overwritten in place.");
}
