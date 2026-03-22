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
    constexpr int N_TILES = 8;

    __shared__ __align__(256) TA shared_memA[WMMA_M * (WMMA_K + PAD) * N_TILES];
    __shared__ __align__(256) TB shared_memB[WMMA_K * (WMMA_N + PAD) * N_TILES];

    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x;



    wmma::fragment<wmma::matrix_a,16,16,16,TA,wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b,16,16,16,TB,wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator,16,16,16,TC> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    #pragma unroll
    for(int k = 0; k < K; k += WMMA_K * N_TILES/2) {

        // initialize shared memory
        if (lane < M * K * 8) {
            shared_memA[lane] = __float2half(1.0f);
        }
        __syncthreads();

        wmma::load_matrix_sync(a_frag, &shared_memA[warp_id * WMMA_M * (WMMA_K + PAD)], WMMA_K + PAD);
        wmma::load_matrix_sync(b_frag, &shared_memB[warp_id * WMMA_K * (WMMA_N + PAD)], WMMA_N + PAD);

        // fake compute to keep everything alive
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    int warps_per_block = blockDim.x / 32;

    int global_warp_id =
        (blockIdx.y * gridDim.x + blockIdx.x) * warps_per_block
        + warp_id;

    int tile_row = blockIdx.y;
    int tile_col = blockIdx.x;

    wmma::store_matrix_sync(
        &out[(tile_row * 16) * N + (tile_col * 16)],
        c_frag,
        WMMA_N,
        wmma::mem_row_major
    );
}

int main() {
 
    constexpr size_t M = 1*1024, N = 1*1024, K = 16*1024;
    dim3 thread_blocks = 256;
    dim3 dim_grid(N / 16, M / 16);
    
    using TA = half;
    using TB = half;
    using TC = float;

    TC* c_out;
    cuda_check(cudaMalloc(&c_out, M * N * sizeof(TC)));

    wmma_test<TA, TB, TC><<<dim_grid, thread_blocks>>>(c_out, M, N, K);
    cuda_check(cudaDeviceSynchronize());
    return 0;
}