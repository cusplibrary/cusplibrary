/*
 *  Copyright 2008-2009 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <stack>

#include <cusp/exception.h>

#include <thrust/fill.h>

namespace cusp
{
namespace graph
{
namespace detail
{
namespace host
{

template<typename MatrixType, typename ArrayType>
size_t connected_components(const MatrixType& G, ArrayType& components)
{
    typedef typename MatrixType::index_type VertexId;

    VertexId num_nodes = G.num_rows;

    thrust::fill(components.begin(), components.begin() + num_nodes, -1);
    std::stack<VertexId> DFS;
    VertexId component = 0;

    for(VertexId i = 0; i < num_nodes; i++)
    {
        if(components[i] == -1)
        {
            DFS.push(i);
            components[i] = component;

            while (!DFS.empty())
            {
                VertexId top = DFS.top();
                DFS.pop();
   
                for(VertexId jj = G.row_offsets[top]; jj < G.row_offsets[top + 1]; jj++){
                    const VertexId j = G.column_indices[jj];
                    if(components[j] == -1){
                        DFS.push(j);
                        components[j] = component;
                    }
                }
            }

            component++;
        }
    }

    return component;
}

} // end namespace host
} // end namespace detail
} // end namespace graph
} // end namespace cusp
