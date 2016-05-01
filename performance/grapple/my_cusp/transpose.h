
#include <cusp/transpose.h>

template <typename MatrixType1, typename MatrixType2>
void transpose(my_policy& exec,
               const MatrixType1& A, MatrixType2& At)
{
    std::cout << "calling my transpose\n";

    using cusp::system::detail::generic::transpose;

    transpose(exec.get(), A, At);
}

