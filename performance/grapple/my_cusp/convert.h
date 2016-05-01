#include <cusp/convert.h>


template <typename MatrixType1,
          typename MatrixType2>
void convert(my_policy& exec,
             const MatrixType1& src,
             MatrixType2& dst)
{
    std::cout << "calling my convert\n";

    using cusp::system::detail::generic::convert;
    convert(exec.get(), src, dst);
}

