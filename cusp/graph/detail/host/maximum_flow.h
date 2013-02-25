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

#include <cusp/exception.h>

namespace cusp
{
namespace graph
{
namespace detail
{
namespace host
{

typedef long long LL;

struct Edge {
    int from, to, cap, flow, index;
    Edge(int from, int to, int cap, int flow, int index) :
        from(from), to(to), cap(cap), flow(flow), index(index) {}
};

class PushRelabel {
    int N;
    std::vector<std::vector<Edge> > G;
    std::vector<LL> excess;
    std::vector<int> dist, active, count;
    std::queue<int> Q;

    void AddEdge(int from, int to, int cap) {
        G[from].push_back(Edge(from, to, cap, 0, G[to].size()));
        if (from == to) G[from].back().index++;
        G[to].push_back(Edge(to, from, 0, 0, G[from].size() - 1));
    }

    void Enqueue(int v) {
        if (!active[v] && excess[v] > 0) {
            active[v] = true;
            Q.push(v);
        }
    }

    void Push(Edge &e) {
        int amt = int(min(excess[e.from], LL(e.cap - e.flow)));
        if (dist[e.from] <= dist[e.to] || amt == 0) return;
        e.flow += amt;
        G[e.to][e.index].flow -= amt;
        excess[e.to] += amt;
        excess[e.from] -= amt;
        Enqueue(e.to);
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
        for (int i = 0; i < G[v].size(); i++)
            if (G[v][i].cap - G[v][i].flow > 0)
                dist[v] = std::min(dist[v], dist[G[v][i].to] + 1);
        count[dist[v]]++;
        Enqueue(v);
    }

    void Discharge(int v) {
        for (int i = 0; excess[v] > 0 && i < G[v].size(); i++) Push(G[v][i]);
        if (excess[v] > 0) {
            if (count[dist[v]] == 1)
                Gap(dist[v]);
            else
                Relabel(v);
        }
    }

public:
    template<typename MatrixType>
    PushRelabel(const MatrixType& graph) : N(graph.num_rows), G(N), excess(N), dist(N), active(N), count(2*N)
    {
        for( int row = 0; row < N; row++ )
            for( int offset = graph.row_offsets[row]; offset < graph.row_offsets[row + 1]; offset++ )
                AddEdge(row, graph.column_indices[offset], graph.values[offset]);
    }

    LL GetMaxFlow(int s, int t) {
        count[0] = N-1;
        count[N] = 1;
        dist[s] = N;
        active[s] = active[t] = true;
        for (int i = 0; i < G[s].size(); i++) {
            excess[s] += G[s][i].cap;
            Push(G[s][i]);
        }

        while (!Q.empty()) {
            int v = Q.front();
            Q.pop();
            active[v] = false;
            Discharge(v);
        }

        LL totflow = 0;
        for (int i = 0; i < G[s].size(); i++) totflow += G[s][i].flow;
        return totflow;
    }

};


template<typename MatrixType, typename IndexType>
typename MatrixType::value_type
maximum_flow(const MatrixType& G, const IndexType src, const IndexType sink)
{
    typedef typename MatrixType::value_type ValueType;
    typedef typename MatrixType::memory_space MemorySpace;

    size_t N = G.num_rows;
    cusp::array1d<LL,cusp::host_memory> excess(N);
    cusp::array1d<int,cusp::host_memory> dist(N), active(N), count(2*N);
    cusp::array1d<ValueType,cusp::host_memory> flow(G.num_entries, 0);
    std::queue<int> Q;

    count[0] = N-1;
    count[N] = 1;
    dist[src] = N;
    active[src] = active[sink] = true;
    for (int i = G.row_offsets[src]; i < G.row_offsets[src + 1]; i++) {
        excess[src] += G.values[i];
        //Push(G.column_indices[i]);
    }

    while (!Q.empty()) {
        int v = Q.front();
        Q.pop();
        active[v] = false;
        //Discharge(v);
    }

    LL totflow = 0;
    for (int i = G.row_offsets[src]; i < G.row_offsets[src + 1]; i++) totflow += flow[i];
    return totflow;
}

} // end namespace host
} // end namespace detail
} // end namespace graph
} // end namespace cusp
