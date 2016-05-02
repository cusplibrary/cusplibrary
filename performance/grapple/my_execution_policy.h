#include <cusp/system/cuda/detail/par.h>

#include <thrust/execution_policy.h>

#include <iostream>

#include "my_thrust_map.h"
#include "my_cusp_map.h"

struct my_policy : public cusp::cuda::execution_policy<my_policy>
{
private:
    typedef thrust::execution_policy<my_policy> UpCastPolicy;

    size_t stack;
    int func_id;
    cudaEvent_t begin;
    cudaEvent_t end;
public:

    my_policy(void) : stack(0), func_id(-1)
    {
        cudaEventCreate(&begin);
        cudaEventCreate(&end);
    }

    ~my_policy(void)
    {
        cudaEventDestroy(begin);
        cudaEventDestroy(end);
    }

    UpCastPolicy& get(void)
    {
        return *this;
    }

    const cusp::system::cuda::detail::par_t& base(void)
    {
        return cusp::cuda::par;
    }

    void start(const size_t id)
    {
        func_id = id;
        stack++;

        cudaEventRecord(begin,0);
    }

    void stop(void)
    {
        float elapsed_time;
        cudaEventRecord(end, 0);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&elapsed_time, begin, end);

        for(size_t k = 0; k < stack - 1; k++) std::cout << "\t";
        std::cout << ARR_THRUST_NAMES[func_id] << "[ " << elapsed_time << " (ms)]\n";
        stack--;
    }
};

#include "my_thrust_func.h"
#include "my_cusp_func.h"

#include "my_cusp/cg.h"

