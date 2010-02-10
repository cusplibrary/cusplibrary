#include <unittest/unittest.h>
#include <cusp/array1d.h>

template <typename MemorySpace>
void TestArray1d(void)
{
    cusp::array1d<int, MemorySpace> a(4);

    ASSERT_EQUAL(a.size(), 4); 

    a[0] = 0;
    a[1] = 1;
    a[2] = 2;
    a[3] = 3;

    a.push_back(4);

    ASSERT_EQUAL(a.size(), 5); 

    ASSERT_EQUAL(a[0], 0);
    ASSERT_EQUAL(a[1], 1);
    ASSERT_EQUAL(a[2], 2);
    ASSERT_EQUAL(a[3], 3);
    ASSERT_EQUAL(a[4], 4);
}
DECLARE_HOST_DEVICE_UNITTEST(TestArray1d);

template <typename MemorySpace>
void TestArray1dEquality(void)
{
    cusp::array1d<int, MemorySpace> A(2);
    A[0] = 10;
    A[1] = 20;
    
    cusp::array1d<int, cusp::host_memory>   h(A.begin(), A.end());
    cusp::array1d<int, cusp::device_memory> d(A.begin(), A.end());
    std::vector<int>                        v(2);
    v[0] = 10;
    v[1] = 20;

    ASSERT_EQUAL_QUIET(A, h);
    ASSERT_EQUAL_QUIET(A, d);
    ASSERT_EQUAL_QUIET(A, v);

    h.push_back(30);
    d.push_back(30);
    v.push_back(30);

    ASSERT_EQUAL_QUIET(A != h, true);
    ASSERT_EQUAL_QUIET(A != d, true);
    ASSERT_EQUAL_QUIET(A != v, true);
}
DECLARE_HOST_DEVICE_UNITTEST(TestArray1dEquality);

