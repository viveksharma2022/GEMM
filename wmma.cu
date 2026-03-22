#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>
#include <cuda_fp16.h>
#include <vector>
#include <type_traits>
#include <iostream>
#include "utils.hpp"
#include <cuda/pipeline>
#include <cuda.h>
#include <cuda/barrier>

using namespace nvcuda;
 
template<typename TA, typename TB, typename TC>
__global__ void wmma_test(TC* __restrict__ out,
    int M, int N, int K) {

    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;
    constexpr int PAD = 8; // to avoid bank conflicts
    constexpr int N_TILES = 16;
    constexpr int NUM_ACCS = 2; // number of accumulators per warp

    __shared__ __align__(256) TA shared_memA[WMMA_M * (WMMA_K + PAD) * N_TILES];
    __shared__ __align__(256) TB shared_memB[WMMA_K * (WMMA_N + PAD) * N_TILES];

    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x;


    wmma::fragment<wmma::matrix_a,16,16,16,TA,wmma::row_major> a_frag[NUM_ACCS];
    wmma::fragment<wmma::matrix_b,16,16,16,TB,wmma::row_major> b_frag[NUM_ACCS];
    wmma::fragment<wmma::accumulator,16,16,16,TC> c_frag[NUM_ACCS];

    #pragma unroll
    for(int i = 0; i < NUM_ACCS; i++) {
        wmma::fill_fragment(c_frag[i], 0.0f);
    }

    // Load the 0th tile for the first accumulator to get things going
    wmma::load_matrix_sync(a_frag[0], &shared_memA[warp_id * WMMA_M * (WMMA_K + PAD)], WMMA_K + PAD);
    wmma::load_matrix_sync(b_frag[0], &shared_memB[warp_id * WMMA_K * (WMMA_N + PAD)], WMMA_N + PAD);


    #pragma unroll
    for(int k = 0; k < K; k += WMMA_K * N_TILES * NUM_ACCS) { // *2 for double buffering

        // initialize shared memory
        if (lane < WMMA_M * WMMA_K * N_TILES) {
            shared_memA[lane] = __float2half(1.0f);
            shared_memB[lane] = __float2half(1.0f);
        }
        __syncthreads();

        // wmma::load_matrix_sync(a_frag, &shared_memA[warp_id * WMMA_M * (WMMA_K + PAD)], WMMA_K + PAD);
        // wmma::load_matrix_sync(b_frag, &shared_memB[warp_id * WMMA_K * (WMMA_N + PAD)], WMMA_N + PAD);

        // // fake compute to keep everything alive
        // wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        
        int i = 0;
        #pragma unroll
        for(; i < NUM_ACCS-1; i++) {

            // compute previous tile while loading next tile (overlap compute with memory ops)
            wmma::mma_sync(c_frag[i], a_frag[i], b_frag[i], c_frag[i]);

            const auto* ptrA = &shared_memA[(warp_id + (i+1) * WMMA_K * N_TILES) * WMMA_M * (WMMA_K + PAD)];
            const auto* ptrB = &shared_memB[(warp_id + (i+1) * WMMA_K * N_TILES) * WMMA_K * (WMMA_N + PAD)];
            
            wmma::load_matrix_sync(a_frag[i+1], ptrA, WMMA_K + PAD);
            wmma::load_matrix_sync(b_frag[i+1], ptrB, WMMA_N + PAD);

        }

        // compute last tile
        wmma::mma_sync(c_frag[i], a_frag[i], b_frag[i], c_frag[i]);

        // auto ptrA0 = &shared_memA[warp_id * WMMA_M * (WMMA_K + PAD)];
        // auto ptrB0 = &shared_memB[warp_id * WMMA_K * (WMMA_N + PAD)];
        // auto ptrA1 = &shared_memA[(warp_id + WMMA_K * N_TILES) * WMMA_M * (WMMA_K + PAD)];
        // auto ptrB1 = &shared_memB[(warp_id + WMMA_K * N_TILES) * WMMA_K * (WMMA_N + PAD)];

        //         // ---- tile 0 ----
        // #pragma unroll

        // wmma::load_matrix_sync(a_frag0, ptrA0, WMMA_K + PAD);
        // wmma::load_matrix_sync(b_frag0, ptrB0, WMMA_N + PAD);

        // // ---- tile 1 ----
        // wmma::load_matrix_sync(a_frag1, ptrA1, WMMA_K + PAD);
        // wmma::load_matrix_sync(b_frag1, ptrB1, WMMA_N + PAD);

        // // compute (independent)
        // wmma::mma_sync(c_frag0, a_frag0, b_frag0, c_frag0);
        // wmma::mma_sync(c_frag1, a_frag1, b_frag1, c_frag1);
    }

    int warps_per_block = blockDim.x / 32;

    int global_warp_id =
        (blockIdx.y * gridDim.x + blockIdx.x) * warps_per_block
        + warp_id;

    int tile_row = blockIdx.y;
    int tile_col = blockIdx.x;

    wmma::store_matrix_sync(
        &out[(tile_row * 16) * N + (tile_col * 16)],
        c_frag[0],
        WMMA_N,
        wmma::mem_row_major
    );
}

int main() {
 
    constexpr size_t M = 1*1024, N = 1*1024, K = 16*1024;
    dim3 thread_blocks = 512;
    dim3 dim_grid(N / 32, M / 16);
    
    using TA = half;
    using TB = half;
    using TC = float;

    TC* c_out;
    cuda_check(cudaMalloc(&c_out, M * N * sizeof(TC)));

    wmma_test<TA, TB, TC><<<dim_grid, thread_blocks>>>(c_out, M, N, K);
    cuda_check(cudaDeviceSynchronize());
    return 0;
}