#include <thrust/execution_policy.h>

#include <iostream>

struct my_policy : public cusp::cuda::execution_policy<my_policy>
{
private:
    typedef thrust::execution_policy<my_policy> UpCastPolicy;
public:
    UpCastPolicy& get(void)
    {
        return *this;
    }
};

#include "my_thrust/copy.h"
#include "my_thrust/inner_product.h"
#include "my_thrust/scatter.h"
#include "my_thrust/transform.h"

#include "my_cusp/blas.h"
#include "my_cusp/convert.h"
#include "my_cusp/elementwise.h"
#include "my_cusp/format_utils.h"
#include "my_cusp/multiply.h"
#include "my_cusp/sort.h"
#include "my_cusp/transpose.h"

#include "my_cusp/cg.h"
