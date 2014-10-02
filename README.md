<hr>
<h3>CUSP : A C++ Templated Sparse Matrix Library</h3>

Current release: v0.4.0 (August 30, 2013)

View the project at [CUSP Website](http://cusplibrary.github.io) and the [cusp-users discussion forum](http://groups.google.com/group/cusp-users) for information and questions.

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

<br><hr>
<h3>Stable Releases</h3>

CUSP releases are labeled using version identifiers having three fields: 
 
| Date | Version |
| ---- | ------- |
| 08/30/2013 | [CUSP v0.4.0](https://github.com/sdalton1/cusplibrary/archive/0.4.0.zip) |
| 03/08/2012 | [CUSP v0.3.1](https://github.com/sdalton1/cusplibrary/archive/0.3.1.zip) |
| 02/04/2012 | [CUSP v0.3.0](https://github.com/sdalton1/cusplibrary/archive/0.3.0.zip) |
| 05/30/2011 | [CUSP v0.2.0](https://github.com/sdalton1/cusplibrary/archive/0.2.0.zip) |
| 07/10/2010 | [CUSP v0.1.0](https://github.com/sdalton1/cusplibrary/archive/0.1.0.zip) |


<br><hr>
<h3>Contributors</h3>

CUSP is developed as an open-source project by [NVIDIA Research](http://research.nvidia.com).

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
