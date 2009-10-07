#include <unittest/unittest.h>
#include <cusp/vector.h>

void TestHostVector(void)
{
    cusp::host_vector<int> vec(4);

    ASSERT_EQUAL(vec.size(), 4); 

    vec[0] = 0;
    vec[1] = 1;
    vec[2] = 2;
    vec[3] = 3;

    vec.push_back(4);

    ASSERT_EQUAL(vec.size(), 5); 

    ASSERT_EQUAL(vec[0], 0);
    ASSERT_EQUAL(vec[1], 1);
    ASSERT_EQUAL(vec[2], 2);
    ASSERT_EQUAL(vec[3], 3);
    ASSERT_EQUAL(vec[4], 4);
}
DECLARE_UNITTEST(TestHostVector);


void TestDeviceVector(void)
{
    cusp::device_vector<int> vec(4);

    ASSERT_EQUAL(vec.size(), 4); 

    vec[0] = 0;
    vec[1] = 1;
    vec[2] = 2;
    vec[3] = 3;

    vec.push_back(4);

    ASSERT_EQUAL(vec.size(), 5); 

    ASSERT_EQUAL(vec[0], 0);
    ASSERT_EQUAL(vec[1], 1);
    ASSERT_EQUAL(vec[2], 2);
    ASSERT_EQUAL(vec[3], 3);
    ASSERT_EQUAL(vec[4], 4);
}
DECLARE_UNITTEST(TestDeviceVector);


void TestVector(void)
{
    cusp::vector<int, cusp::host>   h_vec1(4, 13);
    cusp::vector<int, cusp::device> d_vec1(4, 13);
    
    ASSERT_EQUAL(h_vec1, d_vec1);

    h_vec1[0] = d_vec1[0] = 20;

    ASSERT_EQUAL(h_vec1, d_vec1);

    cusp::vector<int, std::allocator<int> >                  h_vec2(h_vec1);
    cusp::vector<int, thrust::device_malloc_allocator<int> > d_vec2(d_vec1);
    
    ASSERT_EQUAL(h_vec1, d_vec2);
    ASSERT_EQUAL(h_vec2, d_vec1);
    ASSERT_EQUAL(h_vec2, d_vec2);
}
DECLARE_UNITTEST(TestVector);


