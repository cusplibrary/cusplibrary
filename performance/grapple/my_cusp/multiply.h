#include <cusp/multiply.h>

template <typename LinearOperator,
          typename MatrixOrVector1,
          typename MatrixOrVector2>
void multiply(my_policy &exec,
              const LinearOperator&  A,
              const MatrixOrVector1& B,
              MatrixOrVector2& C)
{
    std::cout << "calling my multiply\n";

    using cusp::system::detail::generic::multiply;
    multiply(exec.get(), A, B, C);
}

