#include <thrust/inner_product.h>

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputType>
OutputType inner_product(my_policy& exec,
                         InputIterator1 first1,
                         InputIterator1 last1,
                         InputIterator2 first2,
                         OutputType init)
{
    std::cout << "calling my inner_product\n";

    using thrust::system::detail::generic::inner_product;
    return inner_product(exec.get(), first1, last1, first2, init);
} // end inner_product()

template<typename InputIterator1,
         typename InputIterator2,
         typename OutputType,
         typename BinaryFunction1,
         typename BinaryFunction2>
OutputType inner_product(my_policy &exec,
                         InputIterator1 first1,
                         InputIterator1 last1,
                         InputIterator2 first2,
                         OutputType init,
                         BinaryFunction1 binary_op1,
                         BinaryFunction2 binary_op2)
{
    std::cout << "calling my inner_product\n";

    using thrust::system::detail::generic::inner_product;
    return inner_product(exec.get(), first1, last1, first2, init, binary_op1, binary_op2);
} // end inner_product()

