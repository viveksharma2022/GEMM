#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>
#include <cuda_fp16.h>
#include <vector>
#include <type_traits>
#include <iostream>
#include "utils.hpp"
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda/barrier>
#include <cuda_pipeline.h>

using namespace nvcuda;
namespace cg = cooperative_groups; 

template<typename TA, typename TB, typename TC>
__global__ void wmma_test( 
    TA* __restrict__ a,
    TB* __restrict__ b,
    TC* __restrict__ out,
    int M, int N, int K) {

    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;
    constexpr int PAD = 8; // to avoid bank conflicts
    constexpr int N_TILES = 8;
    constexpr int NUM_ACCS = 4; // number of accumulators per warp

    __shared__ __align__(256) TA shared_memA[2][WMMA_M * (WMMA_K + PAD) * N_TILES];
    __shared__ __align__(256) TB shared_memB[2][WMMA_K * (WMMA_N + PAD) * N_TILES];

    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x;

    int tid = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;

    wmma::fragment<wmma::matrix_a,16,16,16,TA,wmma::row_major> a_frag[NUM_ACCS];
    wmma::fragment<wmma::matrix_b,16,16,16,TB,wmma::row_major> b_frag[NUM_ACCS];
    wmma::fragment<wmma::accumulator,16,16,16,TC> c_frag[NUM_ACCS];

    #pragma unroll
    for(int i = 0; i < NUM_ACCS; i++) {
        wmma::fill_fragment(c_frag[i], 0.0f);
    }

    // initialize shared memory.. load the first kth tile 
    int idx = 0;
    if (lane < WMMA_M * WMMA_K * N_TILES) {
        shared_memA[idx%2][lane] = a[tid];
        shared_memB[idx%2][lane] = b[tid];
    }
    __syncthreads();

    // Pipeline setup
    __shared__ cuda::pipeline_shared_state<
        cuda::thread_scope_block, 2> pipe_state;

    auto block = cg::this_thread_block();
    auto pipe = cuda::make_pipeline(block, &pipe_state);

    #pragma unroll
    for(int k = 0; k < K; k += WMMA_K * N_TILES * NUM_ACCS) { // *2 for double buffering

        if (lane < WMMA_M * WMMA_K * N_TILES) {
            pipe.producer_acquire();
            cuda::memcpy_async(&shared_memA[(idx+1)%2][lane], &a[tid], sizeof(TA), pipe);
            cuda::memcpy_async(&shared_memB[(idx+1)%2][lane], &b[tid], sizeof(TB), pipe);
            pipe.producer_commit();
        }
        
        #pragma unroll
        for(int i = 0; i < NUM_ACCS; i++) {

            const auto* ptrA = &shared_memA[idx %2][(warp_id + (i) * WMMA_K * N_TILES) * WMMA_M * (WMMA_K + PAD)];
            const auto* ptrB = &shared_memB[idx %2][(warp_id + (i) * WMMA_K * N_TILES) * WMMA_K * (WMMA_N + PAD)];

            wmma::load_matrix_sync(a_frag[i], ptrA, WMMA_K + PAD);
            wmma::load_matrix_sync(b_frag[i], ptrB, WMMA_N + PAD);

        }

        #pragma unroll
        for(int i = 0; i < NUM_ACCS; i++)   
            wmma::mma_sync(c_frag[i], a_frag[i], b_frag[i], c_frag[i]);


        pipe.consumer_wait();
        pipe.consumer_release();
        idx++;

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
 
    constexpr size_t M = 1*1024, N = 4*1024, K = 16*1024;
    dim3 thread_blocks = 256;
    dim3 dim_grid(N / 16 , M / 16);
    
    using TA = half;
    using TB = half;
    using TC = float;

    TA* a;
    TB* b;
    TC* c_out;

    cuda_check(cudaMalloc(&a, M * K * sizeof(TA)));
    cuda_check(cudaMalloc(&b, K * N * sizeof(TB )));
    cuda_check(cudaMalloc(&c_out, M * N * sizeof(TC)));

    wmma_test<TA, TB, TC><<<dim_grid, thread_blocks>>>(a, b, c_out, M, N, K);
    cuda_check(cudaDeviceSynchronize());
    return 0;
}