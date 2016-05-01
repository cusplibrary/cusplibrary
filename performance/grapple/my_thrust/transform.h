#include <thrust/transform.h>

template<typename InputIterator,
         typename OutputIterator,
         typename UnaryFunction>
  OutputIterator transform(my_policy &exec,
                           InputIterator first, InputIterator last,
                           OutputIterator result,
                           UnaryFunction op)
{
  std::cout << "calling my transform[1]\n";

  using thrust::system::detail::generic::transform;
  return transform(exec.get(), first, last, result, op);
} // end transform()


template<typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename BinaryFunction>
  OutputIterator transform(my_policy &exec,
                           InputIterator1 first1, InputIterator1 last1,
                           InputIterator2 first2,
                           OutputIterator result,
                           BinaryFunction op)
{
  std::cout << "calling my transform[2]\n";

  using thrust::system::detail::generic::transform;
  return transform(exec.get(), first1, last1, first2, result, op);
} // end transform()


template<typename InputIterator,
         typename ForwardIterator,
         typename UnaryFunction,
         typename Predicate>
  ForwardIterator transform_if(my_policy &exec,
                               InputIterator first, InputIterator last,
                               ForwardIterator result,
                               UnaryFunction op,
                               Predicate pred)
{
  std::cout << "calling my transform_if[1]\n";

  using thrust::system::detail::generic::transform_if;
  return transform_if(exec.get(), first, last, result, op, pred);
} // end transform_if()


template<typename InputIterator1,
         typename InputIterator2,
         typename ForwardIterator,
         typename UnaryFunction,
         typename Predicate>
  ForwardIterator transform_if(my_policy &exec,
                               InputIterator1 first, InputIterator1 last,
                               InputIterator2 stencil,
                               ForwardIterator result,
                               UnaryFunction op,
                               Predicate pred)
{
  std::cout << "calling my transform_if[2]\n";

  using thrust::system::detail::generic::transform_if;
  return transform_if(exec.get(), first, last, stencil, result, op, pred);
} // end transform_if()


__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename InputIterator3,
         typename ForwardIterator,
         typename BinaryFunction,
         typename Predicate>
__host__ __device__
  ForwardIterator transform_if(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                               InputIterator1 first1, InputIterator1 last1,
                               InputIterator2 first2,
                               InputIterator3 stencil,
                               ForwardIterator result,
                               BinaryFunction binary_op,
                               Predicate pred)
{
  using thrust::system::detail::generic::transform_if;
  return transform_if(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first1, last1, first2, stencil, result, binary_op, pred);
} // end transform_if()

