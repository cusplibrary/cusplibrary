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
 * CTA tile-processing abstraction for BFS frontier expansion+contraction
 ******************************************************************************/

#pragma once

#include "../../../util/device_intrinsics.cuh"
#include "../../../util/cta_work_progress.cuh"
#include "../../../util/scan/cooperative_scan.cuh"
#include "../../../util/io/modified_load.cuh"
#include "../../../util/io/modified_store.cuh"
#include "../../../util/io/load_tile.cuh"
#include "../../../util/io/initialize_tile.cuh"
#include "../../../util/operators.cuh"

#include "../../../util/soa_tuple.cuh"
#include "../../../util/scan/soa/cooperative_soa_scan.cuh"

B40C_NS_PREFIX

namespace b40c {
namespace graph {
namespace bfs {
namespace expand_contract_atomic {


/**
 * Templated texture reference for visited mask
 */
template <typename VisitedMask>
struct BitmaskTex
{
	static texture<VisitedMask, cudaTextureType1D, cudaReadModeElementType> ref;
};
template <typename VisitedMask>
texture<VisitedMask, cudaTextureType1D, cudaReadModeElementType> BitmaskTex<VisitedMask>::ref;


/**
 * Templated texture reference for row-offsets
 */
template <typename SizeT>
struct RowOffsetTex
{
	static texture<SizeT, cudaTextureType1D, cudaReadModeElementType> ref;
};
template <typename SizeT>
texture<SizeT, cudaTextureType1D, cudaReadModeElementType> RowOffsetTex<SizeT>::ref;



/**
 * CTA tile-processing abstraction for BFS frontier expansion+contraction
 */
template <typename KernelPolicy>
struct Cta
{
	//---------------------------------------------------------------------
	// Typedefs
	//---------------------------------------------------------------------

	// Row-length cutoff below which we expand neighbors by writing gather
	// offsets into scratch space (instead of gang-pressing warps or the entire CTA)
	static const int SCAN_EXPAND_CUTOFF = B40C_WARP_THREADS(KernelPolicy::CUDA_ARCH);

	typedef typename KernelPolicy::SmemStorage			SmemStorage;
	typedef typename KernelPolicy::VertexId 			VertexId;
	typedef typename KernelPolicy::SizeT 				SizeT;
	typedef typename KernelPolicy::VisitedMask 			VisitedMask;

	typedef typename KernelPolicy::RakingExpandDetails 	RakingExpandDetails;
	typedef typename KernelPolicy::RakingContractDetails 	RakingContractDetails;

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Input and output device pointers
	VertexId 				*d_in;						// Incoming vertex frontier
	VertexId 				*d_out;						// Outgoing vertex frontier
	VertexId				*d_column_indices;			// CSR column-indices array
	SizeT					*d_row_offsets;				// CSR row-offsets array
	VertexId				*d_labels;					// BFS labels to set
	VisitedMask 			*d_visited_mask;			// Mask for detecting visited status

	// Work progress
	VertexId 				iteration;					// Current BFS iteration
	VertexId 				queue_index;				// Current frontier queue counter index
	util::CtaWorkProgress	&work_progress;				// Atomic workstealing and queueing counters
	SizeT					max_vertex_frontier;		// Maximum size (in elements) of vertex frontiers

	// Operational details for raking scan grids
	RakingExpandDetails 		raking_expand_details;
	RakingContractDetails 		raking_contract_details;

	// Shared memory
	SmemStorage 			&smem_storage;

	// Whether or not to perform bitmask culling (incurs extra latency on small frontiers)
	bool 					bitmask_cull;

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	/**
	 * BitmaskCull
	 */
	__device__ __forceinline__ void BitmaskCull(VertexId &neighbor_id)
	{
		if (neighbor_id != -1) {

			// Location of mask byte to read
			SizeT mask_byte_offset = (neighbor_id & KernelPolicy::VERTEX_ID_MASK) >> 3;

			// Bit in mask byte corresponding to current vertex id
			VisitedMask mask_bit = 1 << (neighbor_id & 7);

			// Read byte from from visited mask (tex)
			VisitedMask mask_byte = tex1Dfetch(
				BitmaskTex<VisitedMask>::ref,
				mask_byte_offset);

			if (mask_bit & mask_byte) {

				// Seen it
				neighbor_id = -1;

			} else {

				util::io::ModifiedLoad<util::io::ld::cg>::Ld(
					mask_byte, d_visited_mask + mask_byte_offset);

				if (mask_bit & mask_byte) {

					// Seen it
					neighbor_id = -1;

				} else {

					// Update with best effort
					mask_byte |= mask_bit;
					util::io::ModifiedStore<util::io::st::cg>::St(
						mask_byte,
						d_visited_mask + mask_byte_offset);
				}
			}
		}
	}


	/**
	 * VertexCull
	 */
	__device__ __forceinline__ void VertexCull(
		VertexId &neighbor_id, 			// vertex ID to check.  Set -1 if previously visited.
		VertexId predecessor_id)
	{
		if (neighbor_id != -1) {

			VertexId row_id = neighbor_id & KernelPolicy::VERTEX_ID_MASK;

			// Load label of node
			VertexId label;
			util::io::ModifiedLoad<util::io::ld::cg>::Ld(
				label,
				d_labels + row_id);

			if (label != -1) {

				// Seen it
				neighbor_id = -1;

			} else {

				if (KernelPolicy::MARK_PREDECESSORS) {

					// Update label with predecessor vertex
					util::io::ModifiedStore<util::io::st::cg>::St(
						predecessor_id,
						d_labels + row_id);

				} else {

					// Update label with current iteration
					util::io::ModifiedStore<util::io::st::cg>::St(
						iteration + 1,
						d_labels + row_id);
				}
			}
		}
	}


	/**
	 * CtaCull
	 */
	__device__ __forceinline__ void CtaCull(VertexId &vertex)
	{
		// Hash the node-IDs into smem scratch

		int hash = vertex % SmemStorage::HASH_ELEMENTS;
		bool duplicate = false;

		// Hash the node-IDs into smem scratch
		if (vertex != -1) {
			smem_storage.cta_hashtable[hash] = vertex;
		}

		__syncthreads();

		// Retrieve what vertices "won" at the hash locations. If a
		// different node beat us to this hash cell; we must assume
		// that we may not be a duplicate.  Otherwise assume that
		// we are a duplicate... for now.

		if (vertex != -1) {
			VertexId hashed_node = smem_storage.cta_hashtable[hash];
			duplicate = (hashed_node == vertex);
		}

		__syncthreads();

		// For the possible-duplicates, hash in thread-IDs to select
		// one of the threads to be the unique one
		if (duplicate) {
			smem_storage.cta_hashtable[hash] = threadIdx.x;
		}

		__syncthreads();

		// See if our thread won out amongst everyone with similar node-IDs
		if (duplicate) {
			// If not equal to our tid, we are not an authoritative thread
			// for this node-ID
			if (smem_storage.cta_hashtable[hash] != threadIdx.x) {
				vertex = -1;
			}
		}
	}


	/**
	 * WarpCull
	 */
	__device__ __forceinline__ void WarpCull(VertexId &vertex)
	{
		if (vertex != -1) {

			int warp_id 		= threadIdx.x >> 5;
			int hash 			= vertex & (SmemStorage::WARP_HASH_ELEMENTS - 1);

			smem_storage.warp_hashtable[warp_id][hash] = vertex;
			VertexId retrieved = smem_storage.warp_hashtable[warp_id][hash];

			if (retrieved == vertex) {

				smem_storage.warp_hashtable[warp_id][hash] = threadIdx.x;
				VertexId tid = smem_storage.warp_hashtable[warp_id][hash];
				if (tid != threadIdx.x) {
					vertex = -1;
				}
			}
		}
	}


	//---------------------------------------------------------------------
	// Helper Structures
	//---------------------------------------------------------------------

	/**
	 * Tile
	 */
	template <
		int LOG_LOADS_PER_TILE,
		int LOG_LOAD_VEC_SIZE>
	struct Tile
	{
		//---------------------------------------------------------------------
		// Typedefs and Constants
		//---------------------------------------------------------------------

		enum {
			LOADS_PER_TILE 		= 1 << LOG_LOADS_PER_TILE,
			LOAD_VEC_SIZE 		= 1 << LOG_LOAD_VEC_SIZE
		};

		typedef typename util::VecType<SizeT, 2>::Type Vec2SizeT;


		//---------------------------------------------------------------------
		// Members
		//---------------------------------------------------------------------

		// Dequeued vertex ids
		VertexId 	vertex_id[LOADS_PER_TILE][LOAD_VEC_SIZE];

		// Edge list details
		SizeT		row_offset[LOADS_PER_TILE][LOAD_VEC_SIZE];
		SizeT		row_length[LOADS_PER_TILE][LOAD_VEC_SIZE];
		SizeT		local_ranks[LOADS_PER_TILE][LOAD_VEC_SIZE];
		SizeT		row_progress[LOADS_PER_TILE][LOAD_VEC_SIZE];

		// Temporary state for local culling
		int 		hash[LOADS_PER_TILE][LOAD_VEC_SIZE];			// Hash ids for vertex ids
		bool 		duplicate[LOADS_PER_TILE][LOAD_VEC_SIZE];		// Status as potential duplicate

		SizeT 		fine_count;
		SizeT		progress;


		//---------------------------------------------------------------------
		// Helper Structures
		//---------------------------------------------------------------------

		/**
		 * Iterate next vector element
		 */
		template <int LOAD, int VEC, int dummy = 0>
		struct Iterate
		{
			/**
			 * Init
			 */
			static __device__ __forceinline__ void Init(Tile *tile)
			{
				tile->row_length[LOAD][VEC] = 0;
				tile->row_progress[LOAD][VEC] = 0;

				Iterate<LOAD, VEC + 1>::Init(tile);
			}

			/**
			 * Inspect
			 */
			static __device__ __forceinline__ void Inspect(Cta *cta, Tile *tile)
			{
				if (tile->vertex_id[LOAD][VEC] != -1) {

					// Translate vertex-id into local gpu row-id (currently stride of num_gpu)
					VertexId row_id = tile->vertex_id[LOAD][VEC] & KernelPolicy::VERTEX_ID_MASK;

					// Node is previously unvisited: compute row offset and length
					tile->row_offset[LOAD][VEC] = tex1Dfetch(RowOffsetTex<SizeT>::ref, row_id);
					tile->row_length[LOAD][VEC] = tex1Dfetch(RowOffsetTex<SizeT>::ref, row_id + 1) - tile->row_offset[LOAD][VEC];
				}

				// Next vector element
				Iterate<LOAD, VEC + 1>::Inspect(cta, tile);
			}


			/**
			 * Expand by CTA.  Return true if overflow outgoing queue
			 */
			static __device__ __forceinline__ bool ExpandByCta(Cta *cta, Tile *tile)
			{
				// CTA-based expansion/loading
				while (true) {

					// Vie
					if (tile->row_length[LOAD][VEC] >= KernelPolicy::CTA_GATHER_THRESHOLD) {
						cta->smem_storage.state.cta_comm = threadIdx.x;
					}

					__syncthreads();

					// Check
					int owner = cta->smem_storage.state.cta_comm;
					if (owner == KernelPolicy::THREADS) {
						break;
					}

					if (owner == threadIdx.x) {

						// Got control of the CTA: command it
						cta->smem_storage.state.warp_comm[0][0] = tile->row_offset[LOAD][VEC];										// start
						cta->smem_storage.state.warp_comm[0][2] = tile->row_offset[LOAD][VEC] + tile->row_length[LOAD][VEC];		// oob
						if (KernelPolicy::MARK_PREDECESSORS) {
							cta->smem_storage.state.warp_comm[0][3] = tile->vertex_id[LOAD][VEC];									// predecessor
						}

						// Unset row length
						tile->row_length[LOAD][VEC] = 0;

						// Unset my command
						cta->smem_storage.state.cta_comm = KernelPolicy::THREADS;	// invalid
					}

					__syncthreads();

					SizeT coop_offset 	= cta->smem_storage.state.warp_comm[0][0];
					SizeT coop_oob 		= cta->smem_storage.state.warp_comm[0][2];

					VertexId predecessor_id;
					if (KernelPolicy::MARK_PREDECESSORS) {
						predecessor_id = cta->smem_storage.state.warp_comm[0][3];
					}

					// Repeatedly throw the whole CTA at the adjacency list
					while (coop_offset < coop_oob) {

						// Gather
						VertexId neighbor_id = -1;
						SizeT ranks[1][1] = { {0} };
						if (coop_offset + threadIdx.x < coop_oob) {

							util::io::ModifiedLoad<KernelPolicy::COLUMN_READ_MODIFIER>::Ld(
								neighbor_id, cta->d_column_indices + coop_offset + threadIdx.x);

							// Cull visited vertices and update discovered vertices
							if (cta->bitmask_cull) {
								cta->BitmaskCull(neighbor_id);					// using global visited mask
							}
							cta->VertexCull(neighbor_id, predecessor_id);		// using vertex visitation status (update discovered vertices)

							if (neighbor_id != -1) ranks[0][0] = 1;
						}

						// Scan tile of ranks, using an atomic add to reserve
						// space in the contracted queue, seeding ranks
						util::Sum<SizeT> scan_op;
						SizeT new_queue_offset = util::scan::CooperativeTileScan<1>::ScanTileWithEnqueue(
							cta->raking_contract_details,
							ranks,
							cta->work_progress.GetQueueCounter<SizeT>(cta->queue_index + 1),
							scan_op);

						// Check updated queue offset for overflow due to redundant expansion
						if (new_queue_offset >= cta->max_vertex_frontier) {
							cta->work_progress.SetOverflow<SizeT>();
							return true;
						}

						// Scatter neighbor if valid
						if (neighbor_id != -1) {
							util::io::ModifiedStore<KernelPolicy::QUEUE_WRITE_MODIFIER>::St(
								neighbor_id,
								cta->d_out + ranks[0][0]);
						}

						coop_offset += KernelPolicy::THREADS;
					}
				}

				// Next vector element
				return Iterate<LOAD, VEC + 1>::ExpandByCta(cta, tile);
			}


			/**
			 * Expand by scan
			 */
			static __device__ __forceinline__ void ExpandByScan(Cta *cta, Tile *tile)
			{
				// Attempt to make further progress on this dequeued item's neighbor
				// list if its current offset into local scratch is in range
				SizeT scratch_offset = tile->local_ranks[LOAD][VEC] + tile->row_progress[LOAD][VEC] - tile->progress;

				while ((tile->row_progress[LOAD][VEC] < tile->row_length[LOAD][VEC]) &&
					(scratch_offset < SmemStorage::OFFSET_ELEMENTS))
				{
					// Put gather offset into scratch space
					cta->smem_storage.offset_scratch[scratch_offset] = tile->row_offset[LOAD][VEC] + tile->row_progress[LOAD][VEC];

					if (KernelPolicy::MARK_PREDECESSORS) {
						// Put dequeued vertex as the predecessor into scratch space
						cta->smem_storage.predecessor_scratch[scratch_offset] = tile->vertex_id[LOAD][VEC];
					}

					tile->row_progress[LOAD][VEC]++;
					scratch_offset++;
				}

				// Next vector element
				Iterate<LOAD, VEC + 1>::ExpandByScan(cta, tile);
			}
		};


		/**
		 * Iterate next load
		 */
		template <int LOAD, int dummy>
		struct Iterate<LOAD, LOAD_VEC_SIZE, dummy>
		{
			/**
			 * Init
			 */
			static __device__ __forceinline__ void Init(Tile *tile)
			{
				Iterate<LOAD + 1, 0>::Init(tile);
			}

			/**
			 * Inspect
			 */
			static __device__ __forceinline__ void Inspect(Cta *cta, Tile *tile)
			{
				Iterate<LOAD + 1, 0>::Inspect(cta, tile);
			}

			/**
			 * Expand by CTA
			 */
			static __device__ __forceinline__ bool ExpandByCta(Cta *cta, Tile *tile)
			{
				return Iterate<LOAD + 1, 0>::ExpandByCta(cta, tile);
			}

			/**
			 * Expand by scan
			 */
			static __device__ __forceinline__ void ExpandByScan(Cta *cta, Tile *tile)
			{
				Iterate<LOAD + 1, 0>::ExpandByScan(cta, tile);
			}
		};

		/**
		 * Terminate
		 */
		template <int dummy>
		struct Iterate<LOADS_PER_TILE, 0, dummy>
		{
			// Init
			static __device__ __forceinline__ void Init(Tile *tile) {}

			// Inspect
			static __device__ __forceinline__ void Inspect(Cta *cta, Tile *tile) {}

			// ExpandByCta
			static __device__ __forceinline__ bool ExpandByCta(Cta *cta, Tile *tile)
			{
				return false;
			}

			// ExpandByScan
			static __device__ __forceinline__ void ExpandByScan(Cta *cta, Tile *tile) {}
		};


		//---------------------------------------------------------------------
		// Interface
		//---------------------------------------------------------------------

		/**
		 * Constructor
		 */
		__device__ __forceinline__ Tile()
		{
			Iterate<0, 0>::Init(this);
		}

		/**
		 * Inspect dequeued vertices, updating label if necessary and
		 * obtaining edge-list details
		 */
		__device__ __forceinline__ void Inspect(Cta *cta)
		{
			Iterate<0, 0>::Inspect(cta, this);
		}

		/**
		 * Expands neighbor lists for valid vertices at CTA-expansion
		 * granularity.  Return true if overflowed outgoing queue.
		 */
		__device__ __forceinline__ bool ExpandByCta(Cta *cta)
		{
			return Iterate<0, 0>::ExpandByCta(cta, this);
		}

		/**
		 * Expands neighbor lists by local scan rank
		 */
		__device__ __forceinline__ void ExpandByScan(Cta *cta)
		{
			Iterate<0, 0>::ExpandByScan(cta, this);
		}
	};


	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Constructor
	 */
	__device__ __forceinline__ Cta(
		VertexId 				iteration,
		VertexId 				queue_index,
		SmemStorage 			&smem_storage,
		VertexId 				*d_in,
		VertexId 				*d_out,
		VertexId 				*d_column_indices,
		SizeT 					*d_row_offsets,
		VertexId 				*d_labels,
		VisitedMask 			*d_visited_mask,
		util::CtaWorkProgress	&work_progress,
		SizeT					max_vertex_frontier) :

			iteration(iteration),
			queue_index(queue_index),
			raking_expand_details(
				smem_storage.expand_raking_elements,
				smem_storage.state.warpscan,
				0),
			raking_contract_details(
				smem_storage.state.contract_raking_elements,
				smem_storage.state.warpscan,
				0),
			smem_storage(smem_storage),
			d_in(d_in),
			d_out(d_out),
			d_column_indices(d_column_indices),
			d_row_offsets(d_row_offsets),
			d_labels(d_labels),
			d_visited_mask(d_visited_mask),
			work_progress(work_progress),
			max_vertex_frontier(max_vertex_frontier),
			bitmask_cull(
				(KernelPolicy::END_BITMASK_CULL < 0) ?
					true : 														// always bitmask cull
					(KernelPolicy::END_BITMASK_CULL == 0) ?
						false : 												// never bitmask cull
						(iteration < KernelPolicy::END_BITMASK_CULL))
	{
		if (threadIdx.x == 0) {
			smem_storage.state.cta_comm = KernelPolicy::THREADS;		// invalid
		}
	}


	/**
	 * Process a single tile
	 */
	__device__ __forceinline__ void ProcessTile(
		SizeT cta_offset,
		SizeT guarded_elements = KernelPolicy::TILE_ELEMENTS)
	{
		Tile<
			KernelPolicy::LOG_LOADS_PER_TILE,
			KernelPolicy::LOG_LOAD_VEC_SIZE> tile;

		// Load tile
		util::io::LoadTile<
			KernelPolicy::LOG_LOADS_PER_TILE,
			KernelPolicy::LOG_LOAD_VEC_SIZE,
			KernelPolicy::THREADS,
			KernelPolicy::QUEUE_READ_MODIFIER,
			false>::LoadValid(
				tile.vertex_id,
				d_in,
				cta_offset,
				guarded_elements,
				(VertexId) -1);

		// Cull nearby duplicates from the incoming frontier using collision-hashing
//		CtaCull(tile.vertex_id[0][0]);
		WarpCull(tile.vertex_id[0][0]);

		// Inspect dequeued vertices, obtaining edge-list details
		tile.Inspect(this);

		// Enqueue valid edge lists into outgoing queue
		if (tile.ExpandByCta(this)) {
			// overflowed
			return;
		}

		// Copy lengths into ranks
		util::io::InitializeTile<
			KernelPolicy::LOG_LOADS_PER_TILE,
			KernelPolicy::LOG_LOAD_VEC_SIZE,
			KernelPolicy::THREADS>::Copy(tile.local_ranks, tile.row_length);

		// Scan tile of local ranks
		util::Sum<SizeT> scan_op;
		tile.fine_count = util::scan::CooperativeTileScan<KernelPolicy::LOAD_VEC_SIZE>::ScanTile(
			raking_expand_details,
			tile.local_ranks,
			scan_op);

		__syncthreads();


		//
		// Enqueue the adjacency lists of unvisited node-IDs by repeatedly
		// gathering edges into the scratch space, and then
		// having the entire CTA copy the scratch pool into the outgoing
		// frontier queue.
		//

		tile.progress = 0;
		while (tile.progress < tile.fine_count) {

			// Fill the scratch space with gather-offsets for neighbor-lists.
			tile.ExpandByScan(this);

			__syncthreads();

			// Copy scratch space into queue
			int scratch_remainder = B40C_MIN(SmemStorage::OFFSET_ELEMENTS, tile.fine_count - tile.progress);

			for (int scratch_offset = 0;
				scratch_offset < scratch_remainder;
				scratch_offset += KernelPolicy::THREADS)
			{
				// Gather a neighbor
				VertexId neighbor_id = -1;
				SizeT ranks[1][1] = { {0} };
				if (scratch_offset + threadIdx.x < scratch_remainder) {

					util::io::ModifiedLoad<KernelPolicy::COLUMN_READ_MODIFIER>::Ld(
						neighbor_id,
						d_column_indices + smem_storage.offset_scratch[scratch_offset + threadIdx.x]);

					VertexId predecessor_id;
					if (KernelPolicy::MARK_PREDECESSORS) {
						predecessor_id = smem_storage.predecessor_scratch[scratch_offset + threadIdx.x];
					}

					// Cull visited vertices and update discovered vertices
					if (bitmask_cull) {
						BitmaskCull(neighbor_id);					// using global visited mask
					}
					VertexCull(neighbor_id, predecessor_id);		// using vertex visitation status (update discovered vertices)

					if (neighbor_id != -1) ranks[0][0] = 1;
				}

				// Scan tile of ranks, using an atomic add to reserve
				// space in the contracted queue, seeding ranks
				util::Sum<SizeT> scan_op;
				SizeT new_queue_offset = util::scan::CooperativeTileScan<1>::ScanTileWithEnqueue(
					raking_contract_details,
					ranks,
					work_progress.GetQueueCounter<SizeT>(queue_index + 1),
					scan_op);

				// Check updated queue offset for overflow due to redundant expansion
				if (new_queue_offset >= max_vertex_frontier) {
					work_progress.SetOverflow<SizeT>();
					return;
				}

				if (neighbor_id != -1) {

					// Scatter it into queue
					util::io::ModifiedStore<KernelPolicy::QUEUE_WRITE_MODIFIER>::St(
						neighbor_id,
						d_out + ranks[0][0]);
				}
			}

			tile.progress += SmemStorage::OFFSET_ELEMENTS;

			__syncthreads();
		}
	}
};



} // namespace expand_contract_atomic
} // namespace bfs
} // namespace graph
} // namespace b40c

B40C_NS_POSTFIX

