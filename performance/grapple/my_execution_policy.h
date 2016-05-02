#include <cusp/system/cuda/detail/par.h>

#include <thrust/execution_policy.h>

#include <iostream>

#include "my_thrust_map.h"

struct my_policy : public cusp::cuda::execution_policy<my_policy>
{
private:
    typedef thrust::execution_policy<my_policy> UpCastPolicy;

    size_t stack;
public:

    my_policy(void) : stack(0) {}

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
        for(size_t k = 0; k < stack; k++) std::cout << "\t";
        std::cout << ARR_THRUST_NAMES[id] << "\n";
        stack++;
    }

    void stop(void)
    {
        stack--;
    }
};

#include "my_thrust_func.h"

#include "my_cusp/blas.h"
#include "my_cusp/convert.h"
#include "my_cusp/elementwise.h"
#include "my_cusp/format_utils.h"
#include "my_cusp/multiply.h"
#include "my_cusp/sort.h"
#include "my_cusp/transpose.h"

#include "my_cusp/cg.h"
