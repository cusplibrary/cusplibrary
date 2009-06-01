#include <unittest/unittest.h>
#include <cusp/memory.h>

//////////////
// Allocate //
//////////////

void TestNewArray(void)
{
    int * v;
    
    v = cusp::new_array<int, cusp::host_memory>(3);
    cusp::delete_host_array(v);
    
    v = cusp::new_array<int>(3, cusp::host_memory());
    cusp::delete_host_array(v);
    
    v = cusp::new_array<int, cusp::device_memory>(3);
    cusp::delete_device_array(v);
    
    v = cusp::new_array<int>(3, cusp::device_memory());
    cusp::delete_device_array(v);
}
DECLARE_UNITTEST(TestNewArray);


void TestNewHostArray(void)
{
    int * v = cusp::new_host_array<int>(3);
    cusp::delete_host_array(v);
}
DECLARE_UNITTEST(TestNewHostArray);


void TestNewDeviceArray(void)
{
    int * v = cusp::new_device_array<int>(3);
    cusp::delete_device_array(v);
}
DECLARE_UNITTEST(TestNewDeviceArray);


////////////
// Memcpy //
////////////

void TestMemcpyOnHost(void)
{
    int * src = cusp::new_host_array<int>(3);
    int * dst = cusp::new_host_array<int>(3);

    src[0] = 0; src[1] = 1; src[2] = 2;

    cusp::memcpy_on_host(dst, src, 3);

    ASSERT_EQUAL_RANGES(dst, dst + 3, src);

    cusp::delete_host_array(src);
    cusp::delete_host_array(dst);
}
DECLARE_UNITTEST(TestMemcpyOnHost);


void TestMemcpyOnDevice(void)
{
    int * src = cusp::new_device_array<int>(3);
    int * dst = cusp::new_device_array<int>(3);

    cusp::set_device_array_element(src, 0, 0);
    cusp::set_device_array_element(src, 1, 1);
    cusp::set_device_array_element(src, 2, 2);

    cusp::memcpy_on_device(dst, src, 3);
    
    ASSERT_EQUAL(cusp::get_device_array_element(dst, 0), 0);
    ASSERT_EQUAL(cusp::get_device_array_element(dst, 1), 1);
    ASSERT_EQUAL(cusp::get_device_array_element(dst, 2), 2);

    cusp::delete_device_array(src);
    cusp::delete_device_array(dst);
}
DECLARE_UNITTEST(TestMemcpyOnDevice);


void TestMemcpyToHost(void)
{
    int * src = cusp::new_device_array<int>(3);
    int * dst = cusp::new_host_array<int>(3);
    
    cusp::set_device_array_element(src, 0, 0);
    cusp::set_device_array_element(src, 1, 1);
    cusp::set_device_array_element(src, 2, 2);

    cusp::memcpy_to_host(dst, src, 3);
    
    ASSERT_EQUAL(dst[0], 0);
    ASSERT_EQUAL(dst[1], 1);
    ASSERT_EQUAL(dst[2], 2);

    cusp::delete_device_array(src);
    cusp::delete_host_array(dst);
}
DECLARE_UNITTEST(TestMemcpyToHost);


void TestMemcpyToDevice(void)
{
    int * src = cusp::new_host_array<int>(3);
    int * dst = cusp::new_device_array<int>(3);
    
    src[0] = 0;
    src[1] = 1;
    src[2] = 2;

    cusp::memcpy_to_device(dst, src, 3);
    
    ASSERT_EQUAL(cusp::get_device_array_element(dst, 0), 0);
    ASSERT_EQUAL(cusp::get_device_array_element(dst, 1), 1);
    ASSERT_EQUAL(cusp::get_device_array_element(dst, 2), 2);

    cusp::delete_host_array(src);
    cusp::delete_device_array(dst);
}
DECLARE_UNITTEST(TestMemcpyToDevice);


/////////////////////
// Duplicate Array //
/////////////////////

void TestDuplicateArrayOnHost(void)
{
    int * src = cusp::new_host_array<int>(3);

    src[0] = 0; src[1] = 1; src[2] = 2;

    int * dst = cusp::duplicate_array_on_host(src, 3);

    ASSERT_EQUAL_RANGES(dst, dst + 3, src);

    cusp::delete_host_array(src);
    cusp::delete_host_array(dst);
}
DECLARE_UNITTEST(TestDuplicateArrayOnHost);

void TestDuplicateArrayOnDevice(void)
{
    int * src = cusp::new_device_array<int>(3);

    cusp::set_device_array_element(src, 0, 0);
    cusp::set_device_array_element(src, 1, 1);
    cusp::set_device_array_element(src, 2, 2);

    int * dst = cusp::duplicate_array_on_device(src, 3);

    ASSERT_EQUAL(cusp::get_device_array_element(dst, 0), 0);
    ASSERT_EQUAL(cusp::get_device_array_element(dst, 1), 1);
    ASSERT_EQUAL(cusp::get_device_array_element(dst, 2), 2);

    cusp::delete_device_array(src);
    cusp::delete_device_array(dst);
}
DECLARE_UNITTEST(TestDuplicateArrayOnDevice);

void TestDuplicateArrayToHost(void)
{
    int * src = cusp::new_device_array<int>(3);
    
    cusp::set_device_array_element(src, 0, 0);
    cusp::set_device_array_element(src, 1, 1);
    cusp::set_device_array_element(src, 2, 2);
    
    int * dst = cusp::duplicate_array_to_host(src, 3);

    ASSERT_EQUAL(dst[0], 0);
    ASSERT_EQUAL(dst[1], 1);
    ASSERT_EQUAL(dst[2], 2);

    cusp::delete_device_array(src);
    cusp::delete_host_array(dst);
}
DECLARE_UNITTEST(TestDuplicateArrayToHost);

void TestDuplicateArrayToDevice(void)
{
    int * src = cusp::new_host_array<int>(3);
    
    src[0] = 0;
    src[1] = 1;
    src[2] = 2;

    int * dst = cusp::duplicate_array_to_device(src, 3);
    
    ASSERT_EQUAL(cusp::get_device_array_element(dst, 0), 0);
    ASSERT_EQUAL(cusp::get_device_array_element(dst, 1), 1);
    ASSERT_EQUAL(cusp::get_device_array_element(dst, 2), 2);

    cusp::delete_host_array(src);
    cusp::delete_device_array(dst);
}
DECLARE_UNITTEST(TestDuplicateArrayToDevice);


