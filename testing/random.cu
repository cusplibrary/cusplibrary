#include <unittest/unittest.h>

#include <cusp/detail/random.h>

#include <cusp/array1d.h>
#include <cusp/array2d.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/extrema.h>

#include <limits>

template <typename T>
struct TestRandomIntegersProbability
{
    void operator()(void)
    {
        size_t n = 123456;
        cusp::detail::random_integers<T> random(n);

        cusp::array2d<size_t, cusp::host_memory> counts(2 * sizeof(T), 16, 0);

        for (size_t i = 0; i < n; i++)
        {
            unsigned long long raw = random[i] - std::numeric_limits<T>::min();
            for (size_t nibble = 0; nibble < 2 * sizeof(T); nibble++)
            {
                counts(nibble, (raw >> (4 * nibble)) % 16)++;
            }
        }
        
        //std::cout << "min " << *thrust::min_element(counts.values.begin(), counts.values.end()) << std::endl;
        //std::cout << "max " << *thrust::max_element(counts.values.begin(), counts.values.end()) << std::endl;
        //cusp::print_matrix(counts);

        size_t expected = n / 16;
        size_t min_bin = *thrust::min_element(counts.values.begin(), counts.values.end());
        size_t max_bin = *thrust::max_element(counts.values.begin(), counts.values.end());
        
        ASSERT_GEQUAL(min_bin, (size_t) (0.95 * expected));
        ASSERT_LEQUAL(max_bin, (size_t) (1.05 * expected));
    }
};
SimpleUnitTest<TestRandomIntegersProbability, IntegralTypes> TestRandomIntegersProbabilityInstance;


template <typename T>
struct TestRandomIntegers
{
    void operator()(void)
    {
        size_t n = 12345;
        cusp::detail::random_integers<T> random(n);

        cusp::array1d<T, cusp::host_memory>   h(random);
        cusp::array1d<T, cusp::device_memory> d(random);

        ASSERT_EQUAL(h, d);
    }
};
SimpleUnitTest<TestRandomIntegers, IntegralTypes> TestRandomIntegersInstance;

