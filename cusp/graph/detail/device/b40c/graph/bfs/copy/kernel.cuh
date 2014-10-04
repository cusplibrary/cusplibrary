/******************************************************************************
 * Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2013, NVIDIA CORPORATION.  All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/******************************************************************************
 * Upsweep BFS Copy kernel
 ******************************************************************************/

#pragma once

#include <b40c/util/cta_work_distribution.cuh>
#include <b40c/util/cta_work_progress.cuh>
#include <b40c/util/kernel_runtime_stats.cuh>

#include <b40c/graph/bfs/copy/cta.cuh>

namespace b40c {
namespace graph {
namespace bfs {
namespace copy {


/**
 * Expansion pass (non-workstealing)
 */
template <typename KernelPolicy, bool WORK_STEALING>
struct SweepPass
{
	static __device__ __forceinline__ void Invoke(
		typename KernelPolicy::VertexId 		&iteration,
		typename KernelPolicy::VertexId 		&steal_index,
		int 									&num_gpus,
		typename KernelPolicy::VertexId 		*&d_in,
		typename KernelPolicy::VertexId 		*&d_out,
		typename KernelPolicy::VertexId 		*&d_predecessor_in,
		typename KernelPolicy::VertexId			*&d_labels,
		util::CtaWorkProgress 					&work_progress,
		util::CtaWorkDistribution<typename KernelPolicy::SizeT> &work_decomposition)
	{
		typedef Cta<KernelPolicy> 					Cta;
		typedef typename KernelPolicy::SizeT 		SizeT;

		// Determine our threadblock's work range
		util::CtaWorkLimits<SizeT> work_limits;
		work_decomposition.template GetCtaWorkLimits<
			KernelPolicy::LOG_TILE_ELEMENTS,
			KernelPolicy::LOG_SCHEDULE_GRANULARITY>(work_limits);

		// Return if we have no work to do
		if (!work_limits.elements) {
			return;
		}

		// CTA processing abstraction
		Cta cta(
			iteration,
			num_gpus,
			d_in,
			d_out,
			d_predecessor_in,
			d_labels);

		// Process full tiles
		while (work_limits.offset < work_limits.guarded_offset) {

			cta.ProcessTile(work_limits.offset);
			work_limits.offset += KernelPolicy::TILE_ELEMENTS;
		}

		// Clean up last partial tile with guarded-i/o
		if (work_limits.guarded_elements) {
			cta.ProcessTile(
				work_limits.offset,
				work_limits.guarded_elements);
		}
	}
};


template <typename SizeT, typename StealIndex>
__device__ __forceinline__ SizeT StealWork(
	util::CtaWorkProgress &work_progress,
	int count,
	StealIndex steal_index)
{
	__shared__ SizeT s_offset;		// The offset at which this CTA performs tile processing, shared by all

	// Thread zero atomically steals work from the progress counter
	if (threadIdx.x == 0) {
		s_offset = work_progress.Steal<SizeT>(count, steal_index);
	}

	__syncthreads();		// Protect offset

	return s_offset;
}



/**
 * Expansion pass (workstealing)
 */
template <typename KernelPolicy>
struct SweepPass <KernelPolicy, true>
{
	static __device__ __forceinline__ void Invoke(
		typename KernelPolicy::VertexId 		&iteration,
		typename KernelPolicy::VertexId 		&steal_index,
		int 									&num_gpus,
		typename KernelPolicy::VertexId 		*&d_in,
		typename KernelPolicy::VertexId 		*&d_out,
		typename KernelPolicy::VertexId 		*&d_predecessor_in,
		typename KernelPolicy::VertexId			*&d_labels,
		util::CtaWorkProgress 					&work_progress,
		util::CtaWorkDistribution<typename KernelPolicy::SizeT> &work_decomposition)
	{
		typedef Cta<KernelPolicy> 					Cta;
		typedef typename KernelPolicy::SizeT 		SizeT;

		// CTA processing abstraction
		Cta cta(
			iteration,
			num_gpus,
			d_in,
			d_out,
			d_predecessor_in,
			d_labels);

		// Total number of elements in full tiles
		SizeT unguarded_elements = work_decomposition.num_elements & (~(KernelPolicy::TILE_ELEMENTS - 1));

		// Worksteal full tiles, if any
		SizeT offset;
		while ((offset = StealWork<SizeT>(work_progress, KernelPolicy::TILE_ELEMENTS, steal_index)) < unguarded_elements) {
			cta.ProcessTile(offset);
		}

		// Last CTA does any extra, guarded work (first tile seen)
		if (blockIdx.x == gridDim.x - 1) {
			SizeT guarded_elements = work_decomposition.num_elements - unguarded_elements;
			cta.ProcessTile(unguarded_elements, guarded_elements);
		}
	}
};


/******************************************************************************
 * Copy Kernel Entrypoint
 ******************************************************************************/

/**
 * Copy kernel entry point
 */
template <typename KernelPolicy>
__launch_bounds__ (KernelPolicy::THREADS, KernelPolicy::CTA_OCCUPANCY)
__global__
void Kernel(
	typename KernelPolicy::VertexId 		iteration,
	typename KernelPolicy::SizeT			num_elements,
	typename KernelPolicy::VertexId 		queue_index,
	typename KernelPolicy::VertexId 		steal_index,
	int										num_gpus,
	typename KernelPolicy::VertexId 		*d_in,
	typename KernelPolicy::VertexId 		*d_out,
	typename KernelPolicy::VertexId 		*d_predecessor_in,
	typename KernelPolicy::VertexId			*d_labels,
	util::CtaWorkProgress 					work_progress,
	util::KernelRuntimeStats				kernel_stats)
{
#if __B40C_CUDA_ARCH__ >= 200

	typedef typename KernelPolicy::SizeT SizeT;

	__shared__ util::CtaWorkDistribution<SizeT> work_decomposition;

	if (KernelPolicy::INSTRUMENT && (threadIdx.x == 0)) {
		kernel_stats.MarkStart();
	}

	// Determine work decomposition
	if (threadIdx.x == 0) {

		// Obtain problem size
		if (KernelPolicy::DEQUEUE_PROBLEM_SIZE) {
			num_elements = work_progress.template LoadQueueLength<SizeT>(queue_index);
		}

		// Initialize work decomposition in smem
		work_decomposition.template Init<KernelPolicy::LOG_SCHEDULE_GRANULARITY>(
			num_elements, gridDim.x);

		// Reset our next outgoing queue counter to zero
		work_progress.template StoreQueueLength<SizeT>(0, queue_index + 2);

		// Reset our next workstealing counter to zero
		work_progress.template PrepResetSteal<SizeT>(steal_index + 1);
	}

	// Barrier to protect work decomposition
	__syncthreads();

	SweepPass<KernelPolicy, KernelPolicy::WORK_STEALING>::Invoke(
		iteration,
		steal_index,
		num_gpus,
		d_in,
		d_out,
		d_predecessor_in,
		d_labels,
		work_progress,
		work_decomposition);

	// Enqueue copied amount
	if ((blockIdx.x == 0) && (threadIdx.x == 0)) {
		SizeT outgoing_length = work_progress.template LoadQueueLength<SizeT>(queue_index + 1);
		work_progress.template StoreQueueLength<SizeT>(outgoing_length + num_elements, queue_index + 1);
	}

	if (KernelPolicy::INSTRUMENT && (threadIdx.x == 0)) {
		kernel_stats.MarkStop();
		kernel_stats.Flush();
	}

#endif
}


} // namespace copy
} // namespace bfs
} // namespace graph
} // namespace b40c

