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

#include <cmath>
#include <vector>
#include <iostream>
#include <queue>

#include <thrust/fill.h>
#include <thrust/replace.h>

#include <cusp/exception.h>

namespace cusp
{
namespace graph
{
namespace detail
{
namespace host
{

using namespace thrust::placeholders;

template<typename VertexId, typename Value>
class PushRelabel {

    int N;
    
    const cusp::csr_matrix<VertexId,Value,cusp::host_memory>& G;
    cusp::array1d<Value,cusp::host_memory>& flow;

    cusp::array1d<Value, cusp::host_memory> excess;
    cusp::array1d<int, cusp::host_memory> dist, active, count;
    std::queue<int> Q;

    void Enqueue(int v) {
        if (!active[v] && excess[v] > 0) {
            active[v] = true;
            Q.push(v);
        }
    }

    void Push(int e) { // e is the edge offset
	// TODO : This impacts performance significantly, one binary search per Push operation
	VertexId from = thrust::upper_bound(G.row_offsets.begin(), G.row_offsets.end(), e) - G.row_offsets.begin() - 1;
	VertexId to   = G.column_indices[e];

        int amt = int(min(excess[from], Value(G.values[e] - flow[e])));
        if (dist[from] <= dist[to] || amt == 0) return;
        flow[e] += amt;
	// TODO : Need to map edges to their symmetric partner, iterating is constant overhead
	for(int i = G.row_offsets[to]; i < G.row_offsets[to + 1]; i++)
		if( G.column_indices[i] == from ){ flow[i] -= amt; break; }
        excess[to] += amt;
        excess[from] -= amt;
        Enqueue(to);
    }

    void Gap(int k) {
        for (int v = 0; v < N; v++) {
            if (dist[v] < k) continue;
            count[dist[v]]--;
            dist[v] = max(dist[v], N+1);
            count[dist[v]]++;
            Enqueue(v);
        }
    }

    void Relabel(int v) {
        count[dist[v]]--;
        dist[v] = 2*N;
        for (int i = G.row_offsets[v]; i < G.row_offsets[v + 1]; i++)
            if ((G.values[i] - flow[i]) > 0)
                dist[v] = std::min(dist[v], dist[G.column_indices[i]] + 1);
        count[dist[v]]++;
        Enqueue(v);
    }

    void Discharge(int v) {
        for (int i = G.row_offsets[v]; excess[v] > 0 && i < G.row_offsets[v + 1]; i++) Push(i);
        if (excess[v] > 0) {
            if (count[dist[v]] == 1)
                Gap(dist[v]);
            else
                Relabel(v);
        }
    }

public:

    template<typename MatrixType, typename ArrayType>
    PushRelabel(const MatrixType& graph, ArrayType& flow) : N(graph.num_rows), G(graph), flow(flow), excess(N,Value(0)), dist(N,0), active(N,0), count(2*N,0)
    {
	// initialize flow
	thrust::fill(flow.begin(), flow.end(), Value(0));
    }

    Value GetMaxFlow(VertexId s, VertexId t) {
        count[0] = N-1;
        count[N] = 1;
        dist[s] = N;
        active[s] = active[t] = true;
        for (int i = G.row_offsets[s]; i < G.row_offsets[s + 1]; i++) {
            excess[s] += G.values[i]; // capacity
            Push(i); // Push to location of the edge in reference
        }

        while (!Q.empty()) {
            int v = Q.front();
            Q.pop();
            active[v] = false;
            Discharge(v);
        }

        Value totflow = 0;
        for (int i = G.row_offsets[s]; i < G.row_offsets[s + 1]; i++) totflow += flow[i];
        return totflow;
    }

};


template<typename MatrixType, typename ArrayType, typename IndexType>
typename MatrixType::value_type
maximum_flow(const MatrixType& G, ArrayType& flow, const IndexType src, const IndexType sink)
{
    typedef typename MatrixType::value_type ValueType;
    typedef typename MatrixType::memory_space MemorySpace;

    PushRelabel<IndexType,ValueType> flow_solver(G, flow);
    ValueType max_flow = flow_solver.GetMaxFlow(src, sink);

    thrust::replace_if(flow.begin(), flow.end(), _1 < ValueType(0), ValueType(0));

    return max_flow;
}

} // end namespace host
} // end namespace detail
} // end namespace graph
} // end namespace cusp
