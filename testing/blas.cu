#include <unittest/unittest.h>

#include <cusp/blas.h>


template <class MemorySpace>
void TestAxpy(void)
{
    float * x = cusp::new_array<float, MemorySpace>(6);
    float * y = cusp::new_array<float, MemorySpace>(6);

    cusp::set_array_element<MemorySpace>(x, 0,  7.0f);
    cusp::set_array_element<MemorySpace>(x, 1,  5.0f);
    cusp::set_array_element<MemorySpace>(x, 2,  4.0f);
    cusp::set_array_element<MemorySpace>(x, 3, -3.0f);
    cusp::set_array_element<MemorySpace>(x, 4,  0.0f);
    cusp::set_array_element<MemorySpace>(x, 5,  4.0f);

    cusp::set_array_element<MemorySpace>(y, 0,  0.0f);
    cusp::set_array_element<MemorySpace>(y, 1, -2.0f);
    cusp::set_array_element<MemorySpace>(y, 2,  0.0f);
    cusp::set_array_element<MemorySpace>(y, 3,  5.0f);
    cusp::set_array_element<MemorySpace>(y, 4,  6.0f);
    cusp::set_array_element<MemorySpace>(y, 5,  1.0f);

    cusp::blas<MemorySpace>::axpy(6, 2.0f, x, y);

    ASSERT_EQUAL(cusp::get_array_element<MemorySpace>(y, 0),  14.0);
    ASSERT_EQUAL(cusp::get_array_element<MemorySpace>(y, 1),   8.0);
    ASSERT_EQUAL(cusp::get_array_element<MemorySpace>(y, 2),   8.0);
    ASSERT_EQUAL(cusp::get_array_element<MemorySpace>(y, 3),  -1.0);
    ASSERT_EQUAL(cusp::get_array_element<MemorySpace>(y, 4),   6.0);
    ASSERT_EQUAL(cusp::get_array_element<MemorySpace>(y, 5),   9.0);

    cusp::delete_array<float, MemorySpace>(x);
    cusp::delete_array<float, MemorySpace>(y);
}
DECLARE_HOST_DEVICE_UNITTEST(TestAxpy);


template <class MemorySpace>
void TestCopy(void)
{
    float * x = cusp::new_array<float, MemorySpace>(6);
    float * y = cusp::new_array<float, MemorySpace>(6);

    cusp::set_array_element<MemorySpace>(x, 0,  7.0f);
    cusp::set_array_element<MemorySpace>(x, 1,  5.0f);
    cusp::set_array_element<MemorySpace>(x, 2,  4.0f);
    cusp::set_array_element<MemorySpace>(x, 3, -3.0f);
    cusp::set_array_element<MemorySpace>(x, 4,  0.0f);
    cusp::set_array_element<MemorySpace>(x, 5,  4.0f);

    cusp::set_array_element<MemorySpace>(y, 0,  0.0f);
    cusp::set_array_element<MemorySpace>(y, 1, -2.0f);
    cusp::set_array_element<MemorySpace>(y, 2,  0.0f);
    cusp::set_array_element<MemorySpace>(y, 3,  5.0f);
    cusp::set_array_element<MemorySpace>(y, 4,  6.0f);
    cusp::set_array_element<MemorySpace>(y, 5,  1.0f);

    cusp::blas<MemorySpace>::copy(6, x, y);

    ASSERT_EQUAL(cusp::get_array_element<MemorySpace>(y, 0),  7.0f);
    ASSERT_EQUAL(cusp::get_array_element<MemorySpace>(y, 1),  5.0f);
    ASSERT_EQUAL(cusp::get_array_element<MemorySpace>(y, 2),  4.0f);
    ASSERT_EQUAL(cusp::get_array_element<MemorySpace>(y, 3), -3.0f);
    ASSERT_EQUAL(cusp::get_array_element<MemorySpace>(y, 4),  0.0f);
    ASSERT_EQUAL(cusp::get_array_element<MemorySpace>(y, 5),  4.0f);

    cusp::delete_array<float, MemorySpace>(x);
    cusp::delete_array<float, MemorySpace>(y);
}
DECLARE_HOST_DEVICE_UNITTEST(TestCopy);


template <class MemorySpace>
void TestDot(void)
{
    float * x = cusp::new_array<float, MemorySpace>(6);
    float * y = cusp::new_array<float, MemorySpace>(6);

    cusp::set_array_element<MemorySpace>(x, 0,  7.0f);
    cusp::set_array_element<MemorySpace>(x, 1,  5.0f);
    cusp::set_array_element<MemorySpace>(x, 2,  4.0f);
    cusp::set_array_element<MemorySpace>(x, 3, -3.0f);
    cusp::set_array_element<MemorySpace>(x, 4,  0.0f);
    cusp::set_array_element<MemorySpace>(x, 5,  4.0f);

    cusp::set_array_element<MemorySpace>(y, 0,  0.0f);
    cusp::set_array_element<MemorySpace>(y, 1, -2.0f);
    cusp::set_array_element<MemorySpace>(y, 2,  0.0f);
    cusp::set_array_element<MemorySpace>(y, 3,  5.0f);
    cusp::set_array_element<MemorySpace>(y, 4,  6.0f);
    cusp::set_array_element<MemorySpace>(y, 5,  1.0f);

    ASSERT_EQUAL(cusp::blas<MemorySpace>::dot(6, x, y), -21.0f);
    
    cusp::delete_array<float, MemorySpace>(x);
    cusp::delete_array<float, MemorySpace>(y);
}
DECLARE_HOST_DEVICE_UNITTEST(TestDot);


template <class MemorySpace>
void TestFill(void)
{
    float * x = cusp::new_array<float, MemorySpace>(6);

    cusp::set_array_element<MemorySpace>(x, 0,  7.0f);
    cusp::set_array_element<MemorySpace>(x, 1,  5.0f);
    cusp::set_array_element<MemorySpace>(x, 2,  4.0f);
    cusp::set_array_element<MemorySpace>(x, 3, -3.0f);
    cusp::set_array_element<MemorySpace>(x, 4,  0.0f);
    cusp::set_array_element<MemorySpace>(x, 5,  4.0f);

    cusp::blas<MemorySpace>::fill(6, 1.0f, x);

    ASSERT_EQUAL(cusp::get_array_element<MemorySpace>(x, 0),  1.0f);
    ASSERT_EQUAL(cusp::get_array_element<MemorySpace>(x, 1),  1.0f);
    ASSERT_EQUAL(cusp::get_array_element<MemorySpace>(x, 2),  1.0f);
    ASSERT_EQUAL(cusp::get_array_element<MemorySpace>(x, 3),  1.0f);
    ASSERT_EQUAL(cusp::get_array_element<MemorySpace>(x, 4),  1.0f);
    ASSERT_EQUAL(cusp::get_array_element<MemorySpace>(x, 5),  1.0f);

    cusp::delete_array<float, MemorySpace>(x);
}
DECLARE_HOST_DEVICE_UNITTEST(TestFill);


template <class MemorySpace>
void TestNrm2(void)
{
    float * x = cusp::new_array<float, MemorySpace>(6);
    
    cusp::set_array_element<MemorySpace>(x, 0,  7.0f);
    cusp::set_array_element<MemorySpace>(x, 1,  5.0f);
    cusp::set_array_element<MemorySpace>(x, 2,  4.0f);
    cusp::set_array_element<MemorySpace>(x, 3, -3.0f);
    cusp::set_array_element<MemorySpace>(x, 4,  0.0f);
    cusp::set_array_element<MemorySpace>(x, 5,  1.0f);

    ASSERT_EQUAL(cusp::blas<MemorySpace>::nrm2(6, x), 10.0f);

    cusp::delete_array<float, MemorySpace>(x);
}
DECLARE_HOST_DEVICE_UNITTEST(TestNrm2);


template <class MemorySpace>
void TestScal(void)
{
    float * x = cusp::new_array<float, MemorySpace>(6);

    cusp::set_array_element<MemorySpace>(x, 0,  7.0f);
    cusp::set_array_element<MemorySpace>(x, 1,  5.0f);
    cusp::set_array_element<MemorySpace>(x, 2,  4.0f);
    cusp::set_array_element<MemorySpace>(x, 3, -3.0f);
    cusp::set_array_element<MemorySpace>(x, 4,  0.0f);
    cusp::set_array_element<MemorySpace>(x, 5,  4.0f);

    cusp::blas<MemorySpace>::scal(6, 2.0f, x);

    ASSERT_EQUAL(cusp::get_array_element<MemorySpace>(x, 0),  14.0);
    ASSERT_EQUAL(cusp::get_array_element<MemorySpace>(x, 1),  10.0);
    ASSERT_EQUAL(cusp::get_array_element<MemorySpace>(x, 2),   8.0);
    ASSERT_EQUAL(cusp::get_array_element<MemorySpace>(x, 3),  -6.0);
    ASSERT_EQUAL(cusp::get_array_element<MemorySpace>(x, 4),   0.0);
    ASSERT_EQUAL(cusp::get_array_element<MemorySpace>(x, 5),   8.0);

    cusp::delete_array<float, MemorySpace>(x);
}
DECLARE_HOST_DEVICE_UNITTEST(TestScal);

