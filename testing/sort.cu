#include <unittest/unittest.h>

#include <cusp/sort.h>

template <class Array>
void InitializeSimpleKeySortTest(Array& unsorted_keys, Array& sorted_keys)
{
    unsorted_keys.resize(7);
    unsorted_keys[0] = 1;
    unsorted_keys[1] = 3;
    unsorted_keys[2] = 6;
    unsorted_keys[3] = 5;
    unsorted_keys[4] = 2;
    unsorted_keys[5] = 0;
    unsorted_keys[6] = 4;

    sorted_keys.resize(7);
    sorted_keys[0] = 0;
    sorted_keys[1] = 1;
    sorted_keys[2] = 2;
    sorted_keys[3] = 3;
    sorted_keys[4] = 4;
    sorted_keys[5] = 5;
    sorted_keys[6] = 6;
}

template <typename ArrayType>
void TestCountingSort(void)
{
    typedef typename ArrayType::template rebind<cusp::host_memory>::type HostArray;

    HostArray unsorted_keys;
    HostArray sorted_keys;

    InitializeSimpleKeySortTest(unsorted_keys, sorted_keys);

    ArrayType keys(unsorted_keys);
    ArrayType skeys(sorted_keys);

    cusp::counting_sort(keys, 0, 6);

    ASSERT_EQUAL(keys, skeys);
}
DECLARE_VECTOR_UNITTEST(TestCountingSort);

