#include <unittest/unittest.h>
#include <cusp/array1d.h>

void TestHostArray1d(void)
{
    cusp::array1d<int,cusp::host_memory> arr(4);

    ASSERT_EQUAL(arr.size(), 4); 

    arr[0] = 0;
    arr[1] = 1;
    arr[2] = 2;
    arr[3] = 3;

    arr.push_back(4);

    ASSERT_EQUAL(arr.size(), 5); 

    ASSERT_EQUAL(arr[0], 0);
    ASSERT_EQUAL(arr[1], 1);
    ASSERT_EQUAL(arr[2], 2);
    ASSERT_EQUAL(arr[3], 3);
    ASSERT_EQUAL(arr[4], 4);
}
DECLARE_UNITTEST(TestHostArray1d);


void TestDeviceArray1d(void)
{
    cusp::array1d<int,cusp::device_memory> arr(4);

    ASSERT_EQUAL(arr.size(), 4); 

    arr[0] = 0;
    arr[1] = 1;
    arr[2] = 2;
    arr[3] = 3;

    arr.push_back(4);

    ASSERT_EQUAL(arr.size(), 5); 

    ASSERT_EQUAL(arr[0], 0);
    ASSERT_EQUAL(arr[1], 1);
    ASSERT_EQUAL(arr[2], 2);
    ASSERT_EQUAL(arr[3], 3);
    ASSERT_EQUAL(arr[4], 4);
}
DECLARE_UNITTEST(TestDeviceArray1d);


void TestArray1d(void)
{
    cusp::array1d<int, cusp::host_memory>   h_arr1(4, 13);
    cusp::array1d<int, cusp::device_memory> d_arr1(4, 13);
    
    ASSERT_EQUAL(h_arr1, d_arr1);

    h_arr1[0] = d_arr1[0] = 20;

    ASSERT_EQUAL(h_arr1, d_arr1);

    cusp::array1d<int, std::allocator<int> >                  h_arr2(h_arr1);
    cusp::array1d<int, thrust::device_malloc_allocator<int> > d_arr2(d_arr1);
    
    ASSERT_EQUAL(h_arr1, d_arr2);
    ASSERT_EQUAL(h_arr2, d_arr1);
    ASSERT_EQUAL(h_arr2, d_arr2);
}
DECLARE_UNITTEST(TestArray1d);


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

