<hr>
<h3>CUSP : A C++ Templated Sparse Matrix Library</h3>

Current release: v0.4.0 (August 30, 2013)

View the project at [CUSP Website](http://cusplibrary.github.com/cusplibrary) and the [cusp-users discussion forum](http://groups.google.com/group/cusp-users) for information and questions.

<br><hr>
<h3>A Simple Example</h3>

```C++
#include <cusp/hyb_matrix.h>
#include <cusp/io/matrix_market.h>
#include <cusp/krylov/cg.h>

int main(void)
{
    // create an empty sparse matrix structure (HYB format)
    cusp::hyb_matrix<int, float, cusp::device_memory> A;

    // load a matrix stored in MatrixMarket format
    cusp::io::read_matrix_market_file(A, "5pt_10x10.mtx");

    // allocate storage for solution (x) and right hand side (b)
    cusp::array1d<float, cusp::device_memory> x(A.num_rows, 0);
    cusp::array1d<float, cusp::device_memory> b(A.num_rows, 1);

    // solve the linear system A * x = b with the Conjugate Gradient method
    cusp::krylov::cg(A, x, b);

    return 0;
}
```

Each thread block uses cub::BlockRadixSort to collectively sort 
its own input segment.  The class is specialized by the 
data type being sorted, by the number of threads per block, by the number of 
keys per thread, and implicitly by the targeted compilation architecture.  

The cub::BlockLoad and cub::BlockStore classes are similarly specialized.    
Furthermore, to provide coalesced accesses to device memory, these primitives are 
configured to access memory using a striped access pattern (where consecutive threads 
simultaneously access consecutive items) and then <em>transpose</em> the keys into 
a [<em>blocked arrangement</em>](index.html#sec4sec3) of elements across threads. 

Once specialized, these classes expose opaque \p TempStorage member types.  
The thread block uses these storage types to statically allocate the union of 
shared memory needed by the thread block.  (Alternatively these storage types 
could be aliased to global memory allocations).

<br><hr>
<h3>Stable Releases</h3>

CUB releases are labeled using version identifiers having three fields: 
 
| Date | Version |
| ---- | ------- |
| 05/23/2014 | [CUB v1.3.2](https://github.com/NVlabs/cub/archive/1.3.2.zip) |
| 04/01/2014 | [CUB v1.2.3](https://github.com/NVlabs/cub/archive/1.2.3.zip) |
| 12/10/2013 | [CUB v1.1.1](https://github.com/NVlabs/cub/archive/1.1.1.zip) |
| 08/08/2013 | [CUB v1.0.1](https://github.com/NVlabs/cub/archive/1.0.1.zip) |
| 03/07/2013 | [CUB v0.9.0](https://github.com/NVlabs/cub/archive/0.9.zip) |


<br><hr>
<h3>Contributors</h3>

CUSP is developed as an open-source project by [NVIDIA Research](http://research.nvidia.com).
The primary developers of Cusp are [Steven Dalton](http://research.nvidia.com/users/steven-dalton).

<br><hr>
<h3>Open Source License</h3>

CUSP is available under the Apache open-source license:

```
Copyright (c) 2008-2014, NVIDIA CORPORATION.  All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
   *  Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
   *  Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
   *  Neither the name of the NVIDIA CORPORATION nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
