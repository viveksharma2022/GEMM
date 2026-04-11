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
#include <cooperative_groups.h>

using namespace nvcuda;
namespace cg = cooperative_groups;

// using vec4f = uint4;
using vec2f = float2;
using vec1f = float;
using T = vec1f; // type for loading 4 half elements at once (64 bits)

constexpr int COMPUTE_TILES = 4; // how many tiles in x and y direction to compute in one block
constexpr int WMMA_SIZE = 16; // size of the WMMA tile
constexpr int size_tile_x = 32; // size of the tile of A and B loaded into shared memory in x direction (must be a multiple of WMMA_SIZE)

__device__ __forceinline__
void async_load_tiles(
    int k_iter,
    __half* shared_A_ptr,
    __half* shared_B_ptr,
    __half* A,
    __half* B,
    int global_row,
    int global_col,
    int M,
    int N,
    int K,
    int shared_mem_size_x,
    int TILE_SIZE,
    cuda::pipeline<cuda::thread_scope_block>& pipe
){

    pipe.producer_acquire();
        for(int tile = 0; tile < (WMMA_SIZE / blockDim.y) * COMPUTE_TILES; tile++){
            int block_row = threadIdx.y + tile * blockDim.y;
            int row = global_row + block_row;
            int col = k_iter + threadIdx.x * TILE_SIZE;

            // A tile
            // if (row < M && col < K) 
            {
                T* dst = reinterpret_cast<T*>(
                    &shared_A_ptr[block_row * shared_mem_size_x + threadIdx.x * TILE_SIZE]
                );
                T* src = reinterpret_cast<T*>(&A[row * K + col]);

                // uint32_t smem_addr = __cvta_generic_to_shared(dst);
                // const void* gmem_ptr = src;

                // asm volatile (
                //     "cp.async.ca.shared.global [%0], [%1], 16;\n"
                //     :
                //     : "r"(smem_addr), "l"(gmem_ptr)
                //     );
                cuda::memcpy_async(dst, src, sizeof(T), pipe);
            }

            // B tile
            row = global_col + block_row;
            // if (row < N && col < K) 
            {
                T* dst = reinterpret_cast<T*>(
                    &shared_B_ptr[block_row * shared_mem_size_x + threadIdx.x * TILE_SIZE]
                );
                T* src = reinterpret_cast<T*>(&B[row * K + col]);
                
                // uint32_t smem_addr = __cvta_generic_to_shared(dst);
                // const void* gmem_ptr = src;

                // asm volatile (
                //     "cp.async.ca.shared.global [%0], [%1], 16;\n"
                //     :
                //     : "r"(smem_addr), "l"(gmem_ptr)
                // );

                cuda::memcpy_async(dst, src, sizeof(T), pipe);
            }
        }

        pipe.producer_commit();
}


void __global__ gemm(__half* __restrict__ A, __half* __restrict__ B, __half* C, size_t M, size_t N, size_t K) {

    constexpr int TILE_SIZE = sizeof(T) / sizeof(__half); // 64bits loaded at once

    // constexpr int PAD = (8 - (size_tile_x * TILE_SIZE) % 8); // padding to avoid bank conflicts in shared memory, ensuring that each row of the tile starts at a different bank
    constexpr int shared_mem_size_x = size_tile_x * TILE_SIZE // size of the shared memory in x direction
        + (8 - (size_tile_x * TILE_SIZE) % 8); //PAD 
    constexpr int size_tile = 16 * COMPUTE_TILES * (shared_mem_size_x); // size of one A or B tile buffer
    
    // Double buffering: allocate two sets of buffers (ping-pong)
    extern __shared__ __half __align__(16) shared[];
    
    __half* shared_A[2] = {shared, shared + size_tile};
    __half* shared_B[2] = {shared + size_tile * 2, shared + size_tile * 3};

    int global_row = blockIdx.y * blockDim.y;
    int global_col = blockIdx.x * blockDim.x;

    // int tid = threadIdx.y * blockDim.x + threadIdx.x; // linear thread id within the block
    int warp_id = (threadIdx.y * blockDim.x + threadIdx.x) / 32; // warp id within the block

    int wmma_blockid_x = warp_id % COMPUTE_TILES; // block id in x direction for WMMA
    int wmma_blockid_y = warp_id / COMPUTE_TILES; 

    wmma::fragment<wmma::accumulator,16,16,16,__half> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    // correct pipeline creation
    // shared pipeline state (REQUIRED)
    __shared__ cuda::pipeline_shared_state<
        cuda::thread_scope_block,
        2> pipe_state;

    // create block-scoped pipeline
    auto block = cg::this_thread_block();
    auto pipe  = cuda::make_pipeline(block, &pipe_state);
    block.sync();

    // Prefetch the first tile into buffer 0
    async_load_tiles(0, shared_A[0], shared_B[0], A, B, global_row, global_col, 
        M, N, K, shared_mem_size_x, TILE_SIZE, pipe);

    int num_k_tiles = K  / (blockDim.x * TILE_SIZE);
    
    // Main double-buffered loop
    for (int k_idx = 0; k_idx < num_k_tiles; k_idx++) {

        int current_buffer = k_idx % 2;
        int next_buffer = (k_idx + 1) % 2;

        // Wait for async copies to complete
        pipe.consumer_wait();
        pipe.consumer_release();
        // __syncthreads();

        // Prefetch next
        if ((k_idx + 1) < num_k_tiles) {
            int next_k_col = (k_idx + 1) * blockDim.x * TILE_SIZE;

            async_load_tiles(next_k_col, shared_A[next_buffer], shared_B[next_buffer],
                A, B, global_row, global_col, M, N, K, shared_mem_size_x, TILE_SIZE, pipe);
        }

        // Compute...
        for(int tile = 0; tile < (blockDim.x * TILE_SIZE) / WMMA_SIZE; tile++){

            wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::row_major> b_frag;
            wmma::load_matrix_sync(a_frag,
                &shared_A[current_buffer][wmma_blockid_x * shared_mem_size_x + tile * WMMA_SIZE],
                shared_mem_size_x);

            wmma::load_matrix_sync(b_frag,
                &shared_B[current_buffer][wmma_blockid_y * shared_mem_size_x + tile * WMMA_SIZE],
                shared_mem_size_x);

            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }

    __syncthreads(); // Final sync before store

    {

        int row = global_row + wmma_blockid_y * WMMA_SIZE;
        int col = global_col + wmma_blockid_x * WMMA_SIZE;

        if (row + 16 <= M && col + 16 <= N) 
            wmma::store_matrix_sync(&C[row * N + col], c_frag, N, 
                wmma::mem_row_major);

    }

}

int main(){

    constexpr size_t M = 1*1024, N = 1*1024, K = 16*1024;

    // Calculate shared memory size for double buffering
    // size_tile = 16 * COMPUTE_TILES * (size_tile_x * TILE_SIZE + PAD)
    // We need 4 buffers: shared_A[0], shared_A[1], shared_B[0], shared_B[1]
    // Allocate with some headroom for alignment and extra synchronization
    constexpr size_t shared_bytes = 48 * 1024; // 48 KB to safely accommodate 4 double-buffered tiles
    cudaFuncSetAttribute(
        gemm,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_bytes
    );
    cudaDeviceProp prop;
    cuda_check(cudaGetDeviceProperties(&prop, 0));


    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Maximum shared memory per block: " << prop.sharedMemPerBlock/1024 << " KB" << std::endl;
    std::cout << "Maximum shared memory per SM: " << prop.sharedMemPerMultiprocessor/1024 << " KB" << std::endl;
    std::cout << "Registers per block: " << prop.regsPerBlock << std::endl;
    std::cout << "warp size: " << prop.warpSize << std::endl;

    // Allocate gpu memory for A, B, C
    __half *d_A, *d_B;
    __half *d_C;
    cuda_check(cudaMalloc(&d_A, M * K * sizeof(__half)));
    cuda_check(cudaMalloc(&d_B, K * N * sizeof(__half)));
    cuda_check(cudaMalloc(&d_C, M * N * sizeof(__half)));

    dim3 blockDim(32, COMPUTE_TILES * COMPUTE_TILES); // 512 threads per block
    dim3 gridDim((N ) / (WMMA_SIZE*COMPUTE_TILES),
         (M) / (WMMA_SIZE*COMPUTE_TILES)); // grid dimensions to cover the entire output matrix C
    gemm<<<gridDim, blockDim, shared_bytes>>>(d_A, d_B, d_C, M, N, K);
    cudaError_t err = cudaGetLastError();
    printf("Error: %s\n", cudaGetErrorString(err));

    cuda_check(cudaDeviceSynchronize());

    // Free gpu memory
    cuda_check(cudaFree(d_A));
    cuda_check(cudaFree(d_B));   
    cuda_check(cudaFree(d_C));

    return 0;

}