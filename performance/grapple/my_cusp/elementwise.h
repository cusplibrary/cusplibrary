#include <cusp/elementwise.h>

template <typename MatrixType1,
          typename MatrixType2,
          typename MatrixType3,
          typename BinaryFunction>
void elementwise(my_policy& exec,
                 const MatrixType1& A,
                 const MatrixType2& B,
                       MatrixType3& C,
                       BinaryFunction op)
{
    std::cout << "calling my elementwise\n";

    using cusp::system::detail::generic::elementwise;

    if(A.num_rows != B.num_rows || A.num_cols != B.num_cols)
        throw cusp::invalid_input_exception("matrix dimensions do not match");

    elementwise(exec.get(), A, B, C, op);
}

