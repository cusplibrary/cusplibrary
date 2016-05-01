#include <thrust/scatter.h>

template<typename InputIterator1,
         typename InputIterator2,
         typename RandomAccessIterator>
  void scatter(my_policy &exec,
               InputIterator1 first,
               InputIterator1 last,
               InputIterator2 map,
               RandomAccessIterator output)
{
  std::cout << "calling my scatter\n";

  using thrust::system::detail::generic::scatter;
  return scatter(exec.get(), first, last, map, output);
} // end scatter()


template<typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename RandomAccessIterator>
  void scatter_if(my_policy &exec,
                  InputIterator1 first,
                  InputIterator1 last,
                  InputIterator2 map,
                  InputIterator3 stencil,
                  RandomAccessIterator output)
{
  std::cout << "calling my scatter_if\n";

  using thrust::system::detail::generic::scatter_if;
  return scatter_if(exec.get(), first, last, map, stencil, output);
} // end scatter_if()

