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
 * Tile-processing functionality for BFS copy kernels
 ******************************************************************************/

#pragma once

#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>
#include <b40c/util/io/load_tile.cuh>
#include <b40c/util/io/store_tile.cuh>

namespace b40c {
namespace graph {
namespace bfs {
namespace copy {


/**
 * Derivation of KernelPolicy that encapsulates tile-processing routines
 */
template <typename KernelPolicy>
struct Cta
{
	//---------------------------------------------------------------------
	// Typedefs
	//---------------------------------------------------------------------

	typedef typename KernelPolicy::VertexId 		VertexId;
	typedef typename KernelPolicy::SizeT 			SizeT;

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Current BFS iteration
	VertexId 				iteration;
	int 					num_gpus;

	// Input and output device pointers
	VertexId 				*d_in;
	VertexId 				*d_out;
	VertexId 				*d_predecessor_in;
	VertexId				*d_labels;

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


		//---------------------------------------------------------------------
		// Members
		//---------------------------------------------------------------------

		// Dequeued vertex ids
		VertexId 	vertex_id[LOADS_PER_TILE][LOAD_VEC_SIZE];
		VertexId 	predecessor_id[LOADS_PER_TILE][LOAD_VEC_SIZE];


		//---------------------------------------------------------------------
		// Helper Structures
		//---------------------------------------------------------------------

		/**
		 * Iterate over vertex id
		 */
		template <int LOAD, int VEC, int dummy = 0>
		struct Iterate
		{
			/**
			 * MarkSource
			 */
			static __device__ __forceinline__ void MarkSource(
				Cta *cta,
				Tile *tile)
			{
				if (tile->vertex_id[LOAD][VEC] != -1) {

					VertexId row_id = (tile->vertex_id[LOAD][VEC] & KernelPolicy::VERTEX_ID_MASK) / cta->num_gpus;

					// Load label of node
					VertexId label;
					util::io::ModifiedLoad<util::io::ld::cg>::Ld(
						label,
						cta->d_labels + row_id);


					if (label != -1) {

						// Seen it
						tile->vertex_id[LOAD][VEC] = -1;

					} else {

						if (KernelPolicy::MARK_PREDECESSORS) {

							// Update label with predecessor vertex
							util::io::ModifiedStore<util::io::st::cg>::St(
								tile->predecessor_id[LOAD][VEC],
								cta->d_labels + row_id);
						} else {

							// Update label with current iteration
							util::io::ModifiedStore<util::io::st::cg>::St(
								cta->iteration,
								cta->d_labels + row_id);
						}
					}
				}

				// Next
				Iterate<LOAD, VEC + 1>::MarkSource(cta, tile);
			}
		};


		/**
		 * Iterate next load
		 */
		template <int LOAD, int dummy>
		struct Iterate<LOAD, LOAD_VEC_SIZE, dummy>
		{
			// MarkSource
			static __device__ __forceinline__ void MarkSource(Cta *cta, Tile *tile)
			{
				Iterate<LOAD + 1, 0>::MarkSource(cta, tile);
			}
		};



		/**
		 * Terminate iteration
		 */
		template <int dummy>
		struct Iterate<LOADS_PER_TILE, 0, dummy>
		{
			// MarkSource
			static __device__ __forceinline__ void MarkSource(Cta *cta, Tile *tile) {}
		};


		//---------------------------------------------------------------------
		// Interface
		//---------------------------------------------------------------------

		/**
		 * MarkSource
		 */
		__device__ __forceinline__ void MarkSource(Cta *cta)
		{
			Iterate<0, 0>::MarkSource(cta, this);
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
		int						num_gpus,
		VertexId 				*d_in,
		VertexId 				*d_out,
		VertexId 				*d_predecessor_in,
		VertexId 				*d_labels) :
			iteration(iteration),
			num_gpus(num_gpus),
			d_in(d_in),
			d_out(d_out),
			d_predecessor_in(d_predecessor_in),
			d_labels(d_labels)
	{}


	/**
	 * Process a single tile
	 *
	 * Each thread copies only the strided values it loads.
	 */
	__device__ __forceinline__ void ProcessTile(
		SizeT cta_offset,
		SizeT guarded_elements = KernelPolicy::TILE_ELEMENTS)
	{
		// Tile of elements
		Tile<KernelPolicy::LOG_LOADS_PER_TILE, KernelPolicy::LOG_LOAD_VEC_SIZE> tile;

		// Load tile of vertices
		util::io::LoadTile<
			KernelPolicy::LOG_LOADS_PER_TILE,
			KernelPolicy::LOG_LOAD_VEC_SIZE,
			KernelPolicy::THREADS,
			KernelPolicy::READ_MODIFIER,
			false>::LoadValid(
				tile.vertex_id,
				d_in,
				cta_offset,
				guarded_elements,
				(VertexId) -1);

		if (KernelPolicy::MARK_PREDECESSORS) {

			// Load tile of predecessors
			util::io::LoadTile<
				KernelPolicy::LOG_LOADS_PER_TILE,
				KernelPolicy::LOG_LOAD_VEC_SIZE,
				KernelPolicy::THREADS,
				KernelPolicy::READ_MODIFIER,
				false>::LoadValid(
					tile.predecessor_id,
					d_predecessor_in,
					cta_offset,
					guarded_elements);
		}

		if (KernelPolicy::LOADS_PER_TILE > 1) __syncthreads();

		// Mark sources
		tile.MarkSource(this);

		// Store tile of vertices
		util::io::StoreTile<
			KernelPolicy::LOG_LOADS_PER_TILE,
			KernelPolicy::LOG_LOAD_VEC_SIZE,
			KernelPolicy::THREADS,
			KernelPolicy::WRITE_MODIFIER,
			false>::Store(
				tile.vertex_id,
				d_out,
				cta_offset,
				guarded_elements);
	}
};

} // namespace copy
} // namespace bfs
} // namespace graph
} // namespace b40c

