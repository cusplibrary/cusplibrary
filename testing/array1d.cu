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


