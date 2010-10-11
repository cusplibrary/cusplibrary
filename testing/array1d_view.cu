#include <unittest/unittest.h>

#include <cusp/array1d.h>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
// take these

template <typename MemorySpace>
void TestArray1dView(void)
{
  cusp::array1d<int, MemorySpace> A(4);
  A[0] = 10; A[1] = 20; A[2] = 30; A[3] = 40;

  typedef typename cusp::array1d<int, MemorySpace>::iterator Iterator;

  cusp::array1d_view<Iterator> V(A.begin(), A.end());

  ASSERT_EQUAL(V.size(),     4);
  ASSERT_EQUAL(V.capacity(), 4);
  ASSERT_EQUAL(V[0], 10);
  ASSERT_EQUAL(V[1], 20);
  ASSERT_EQUAL(V[2], 30);
  ASSERT_EQUAL(V[3], 40);
  ASSERT_EQUAL_QUIET(V.begin(), A.begin());
  ASSERT_EQUAL_QUIET(V.end(),   A.end());

  ASSERT_THROWS(V.resize(5), cusp::not_implemented_exception);
  
  V.resize(3);

  ASSERT_EQUAL(V.size(),     3);
  ASSERT_EQUAL(V.capacity(), 4);
  ASSERT_EQUAL_QUIET(V.begin(), A.begin());
  ASSERT_EQUAL_QUIET(V.end(),   A.begin() + 3);

  V[1] = 17;

  ASSERT_EQUAL(V[1], 17);
  ASSERT_EQUAL(A[1], 17);
}
DECLARE_HOST_DEVICE_UNITTEST(TestArray1dView);

// test view = view, array = view, and view = array (also unequal sizes)

template <typename MemorySpace>
void TestArray1dViewCountingIterator(void)
{
  typedef thrust::counting_iterator<int> Iterator;

  cusp::array1d_view<Iterator> V(Iterator(5), Iterator(9));

  ASSERT_EQUAL(V.size(), 4);
  ASSERT_EQUAL(V[0], 5);
  ASSERT_EQUAL(V[3], 8);
}
DECLARE_HOST_DEVICE_UNITTEST(TestArray1dViewCountingIterator);

template <typename MemorySpace>
void TestArray1dViewZipIterator(void)
{
  cusp::array1d<int, MemorySpace> A(4);
  cusp::array1d<int, MemorySpace> B(4);
  A[0] = 10; A[1] = 20; A[2] = 30; A[3] = 40;
  B[0] = 50; B[1] = 60; B[2] = 70; B[3] = 80;

  typedef typename cusp::array1d<int, MemorySpace>::iterator Iterator;
  typedef typename thrust::tuple<Iterator,Iterator>          IteratorTuple;
  typedef typename thrust::zip_iterator<IteratorTuple>       ZipIterator;
  
  ZipIterator begin = thrust::make_zip_iterator(thrust::make_tuple(A.begin(), B.begin()));

  cusp::array1d_view<ZipIterator> V(begin, begin + 4);

  ASSERT_EQUAL(V.size(), 4);
  ASSERT_EQUAL_QUIET(V[0], thrust::make_tuple(10,50));
  ASSERT_EQUAL_QUIET(V[3], thrust::make_tuple(40,80));
}
DECLARE_HOST_DEVICE_UNITTEST(TestArray1dViewZipIterator);

