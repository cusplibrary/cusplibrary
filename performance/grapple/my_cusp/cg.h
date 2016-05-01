#include <cusp/krylov/cg.h>

template <typename LinearOperator,
          typename VectorType1,
          typename VectorType2,
          typename Monitor,
          typename Preconditioner>
void cg(my_policy& exec,
        const LinearOperator& A,
        VectorType1& x,
        const VectorType2& b,
        Monitor& monitor,
        Preconditioner& M)
{
    std::cout << "calling my cg\n";

    using cusp::krylov::cg_detail::cg;
    cg(exec.get(), A, x, b, monitor, M);
}

