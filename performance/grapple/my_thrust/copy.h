#include <thrust/copy.h>

template<typename InputIterator, typename OutputIterator>
  OutputIterator copy(my_policy &exec,
                      InputIterator first,
                      InputIterator last,
                      OutputIterator result)
{
  std::cout << "calling my thrust::copy\n";

  using thrust::system::detail::generic::copy;
  return copy(exec.get(), first, last, result);
} // end copy()


template<typename InputIterator, typename Size, typename OutputIterator>
  OutputIterator copy_n(my_policy &exec,
                        InputIterator first,
                        Size n,
                        OutputIterator result)
{
  std::cout << "calling my copy_n\n";

  using thrust::system::detail::generic::copy_n;
  return copy_n(exec.get(), first, n, result);
} // end copy_n()

template<typename InputIterator,
         typename OutputIterator,
         typename Predicate>
  OutputIterator copy_if(my_policy &exec,
                         InputIterator first,
                         InputIterator last,
                         OutputIterator result,
                         Predicate pred)
{
  std::cout << "calling my copy_if\n";

  using thrust::system::detail::generic::copy_if;
  return copy_if(exec.get(), first, last, result, pred);
} // end copy_if()


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename Predicate>
  OutputIterator copy_if(my_policy &exec,
                         InputIterator1 first,
                         InputIterator1 last,
                         InputIterator2 stencil,
                         OutputIterator result,
                         Predicate pred)
{
  std::cout << "calling my copy_if\n";

  using thrust::system::detail::generic::copy_if;
  return copy_if(exec.get(), first, last, stencil, result, pred);
} // end copy_if()

