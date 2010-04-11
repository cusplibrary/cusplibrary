#include <unittest/unittest.h>

#include <cusp/monitor.h>

template <typename MemorySpace>
void TestMonitorSimple(void)
{
    cusp::array1d<float,MemorySpace> b(2);
    b[0] = 10;
    b[1] =  0;
    
    cusp::array1d<float,MemorySpace> r(2);
    r[0] = 10;
    r[1] =  0;

    cusp::default_monitor<float> monitor(b, 5, 0.5);

    ASSERT_EQUAL(monitor.finished(r), false);
    ASSERT_EQUAL(monitor.iteration_count(), 0);
    ASSERT_EQUAL(monitor.iteration_limit(), 5);
    ASSERT_EQUAL(monitor.relative_tolerance(), 0.5);

    ++monitor;
    
    ASSERT_EQUAL(monitor.finished(r), false);
    ASSERT_EQUAL(monitor.iteration_count(), 1);

    r[0] = 4;
    
    ASSERT_EQUAL(monitor.finished(r), true);
    ASSERT_EQUAL(monitor.iteration_count(), 1);
    
    r[0] = 6;
    
    ASSERT_EQUAL(monitor.finished(r), false);
    ASSERT_EQUAL(monitor.iteration_count(), 1);
    
    ++monitor;

    ASSERT_EQUAL(monitor.finished(r), false);
    ASSERT_EQUAL(monitor.iteration_count(), 2);
    
    ++monitor;
    ++monitor;

    ASSERT_EQUAL(monitor.finished(r), false);
    ASSERT_EQUAL(monitor.iteration_count(), 4);
    
    ++monitor;

    ASSERT_EQUAL(monitor.finished(r), true);
    ASSERT_EQUAL(monitor.iteration_count(), 5);
}
DECLARE_HOST_DEVICE_UNITTEST(TestMonitorSimple);

