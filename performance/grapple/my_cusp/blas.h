#include <cusp/blas.h>

template <typename ArrayType>
int amax(my_policy &exec,
         const ArrayType& x)
{
    std::cout << "calling my blas::amax\n";

    using cusp::system::detail::generic::blas::amax;
    return amax(exec.get(), x);
}

template <typename ArrayType>
typename cusp::norm_type<typename ArrayType::value_type>::type
asum(my_policy &exec,
     const ArrayType& x)
{
    std::cout << "calling my blas::asum\n";

    using cusp::system::detail::generic::blas::asum;
    return asum(exec.get(), x);
}

template <typename ArrayType1,
          typename ArrayType2,
          typename ScalarType>
void axpy(my_policy& exec,
          const ArrayType1& x,
          ArrayType2& y,
          const ScalarType alpha)
{
    std::cout << "calling my blas::axpy\n";

    using cusp::system::detail::generic::blas::axpy;
    axpy(exec.get(), x, y, alpha);
}

template <typename ArrayType1,
          typename ArrayType2,
          typename ArrayType3,
          typename ScalarType1,
          typename ScalarType2>
void axpby(my_policy &exec,
           const ArrayType1& x,
           const ArrayType2& y,
                 ArrayType3& z,
           const ScalarType1 alpha,
           const ScalarType2 beta)
{
    std::cout << "calling my blas::axpby\n";

    cusp::assert_same_dimensions(x, y, z);

    using cusp::system::detail::generic::blas::axpby;
    axpby(exec.get(), x, y, z, alpha, beta);
}

template <typename ArrayType1,
          typename ArrayType2,
          typename ArrayType3,
          typename ArrayType4,
          typename ScalarType1,
          typename ScalarType2,
          typename ScalarType3>
void axpbypcz(my_policy &exec,
              const ArrayType1& x,
              const ArrayType2& y,
              const ArrayType3& z,
                    ArrayType4& output,
              const ScalarType1 alpha,
              const ScalarType2 beta,
              const ScalarType3 gamma)
{
    std::cout << "calling my blas::axpbypcz\n";

    cusp::assert_same_dimensions(x, y, z, output);

    using cusp::system::detail::generic::blas::axpbypcz;
    axpbypcz(exec.get(), x, y, z, output, alpha, beta, gamma);
}

template <typename ArrayType1,
          typename ArrayType2>
void copy(my_policy& exec,
          const ArrayType1& x,
          ArrayType2& y)
{
    std::cout << "calling my blas::copy\n";

    using cusp::system::detail::generic::blas::copy;
    copy(exec.get(), x, y);
}

template <typename ArrayType1,
          typename ArrayType2,
          typename ArrayType3>
void xmy(my_policy &exec,
         const ArrayType1& x,
         const ArrayType2& y,
               ArrayType3& z)
{
    std::cout << "calling my blas::xmy\n";

    cusp::assert_same_dimensions(x, y, z);

    using cusp::system::detail::generic::blas::xmy;
    xmy(exec.get(), x, y, z);
}

template <typename ArrayType1,
          typename ArrayType2>
typename ArrayType1::value_type
dot(my_policy& exec,
    const ArrayType1& x,
    const ArrayType2& y)
{
    std::cout << "calling my blas::dot\n";

    cusp::assert_same_dimensions(x, y);

    using cusp::system::detail::generic::blas::dot;
    return dotc(exec.get(), x, y);
}

template <typename ArrayType1,
          typename ArrayType2>
typename ArrayType1::value_type
dotc(my_policy& exec,
     const ArrayType1& x,
     const ArrayType2& y)
{
    std::cout << "calling my blas::dotc\n";

    cusp::assert_same_dimensions(x, y);

    using cusp::system::detail::generic::blas::dotc;
    return dotc(exec.get(), x, y);
}

template <typename ArrayType,
          typename ScalarType>
void fill(my_policy &exec,
          ArrayType& x,
          const ScalarType alpha)
{
    std::cout << "calling my blas::fill\n";

    using cusp::system::detail::generic::blas::fill;
    fill(exec.get(), x, alpha);
}

template <typename ArrayType>
typename cusp::norm_type<typename ArrayType::value_type>::type
nrm1(my_policy &exec,
     const ArrayType& x)
{
    std::cout << "calling my blas::nrm1\n";

    using cusp::system::detail::generic::blas::nrm1;
    return nrm1(exec.get(), x);
}

template <typename ArrayType>
typename cusp::norm_type<typename ArrayType::value_type>::type
nrm2(my_policy &exec,
     const ArrayType& x)
{
    std::cout << "calling my blas::nrm2\n";

    using cusp::system::detail::generic::blas::nrm2;
    return nrm2(exec.get(), x);
}

template <typename ArrayType>
typename cusp::norm_type<typename ArrayType::value_type>::type
nrmmax(my_policy &exec,
       const ArrayType& x)
{
    std::cout << "calling my blas::nrmmax\n";

    using cusp::system::detail::generic::blas::nrmmax;
    return nrmmax(exec.get(), x);
}

template <typename ArrayType,
          typename ScalarType>
void scal(my_policy &exec,
                ArrayType& x,
          const ScalarType alpha)
{
    std::cout << "calling my blas::scal\n";

    using cusp::system::detail::generic::blas::scal;
    scal(exec.get(), x, alpha);
}

