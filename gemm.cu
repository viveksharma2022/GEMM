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
using vec4f = float4;
using vec2f = float2;
using vec1f = float;
using T = vec1f; // type for loading 4 half elements at once (64 bits)

constexpr int COMPUTE_TILES = 4; // how many tiles in x and y direction to compute in one block
constexpr int WMMA_SIZE = 16; // size of the WMMA tile
constexpr int size_tile_x = 32; // size of the tile of A and B loaded into shared memory in x direction (must be a multiple of WMMA_SIZE)

void __global__ gemm(__half* __restrict__ A, __half* __restrict__ B, float* C, size_t M, size_t N, size_t K) {

    constexpr int TILE_SIZE = sizeof(T) / sizeof(__half); // 64bits loaded at once
    // constexpr int PAD = 1; // padding to avoid bank conflicts in shared memory

    constexpr int PAD = 8 - (size_tile_x * TILE_SIZE) % 8; // padding to avoid bank conflicts in shared memory, ensuring that each row of the tile starts at a different bank
    constexpr int shared_mem_size_x = size_tile_x * TILE_SIZE + PAD; // size of the shared memory in x direction
    constexpr int size_A = 16 * COMPUTE_TILES * (shared_mem_size_x); // size of the shared memory for A
    extern __shared__ __half __align__(16) shared[];

    __half* shared_A = shared;
    __half* shared_B = shared + size_A;

    // // __shared__ __half __align__(256) shared_A[16 * (32 * TILE_SIZE + PAD)]; // assuming blockDim.x = blockDim.y = 32
    // // __shared__ __half __align__(256) shared_B[16 * (32 * TILE_SIZE + PAD)]; // assuming blockDim.x = blockDim.y = 32

    int global_row = blockIdx.y * blockDim.y;
    int global_col = blockIdx.x * blockDim.x;

    int tid = threadIdx.y * blockDim.x + threadIdx.x; // linear thread id within the block
    int warp_id = tid / 32; // warp id within the block

    int wmma_blockid_x = warp_id % COMPUTE_TILES; // block id in x direction for WMMA
    int wmma_blockid_y = warp_id / COMPUTE_TILES; 

    wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator,16,16,16,float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    //read to shared memory A
    for (int k = 0; k < K/TILE_SIZE; k += blockDim.x) {
        
        // read A matrix tile into shared memory, A is in row-major order
        for(int tile = 0; tile < (WMMA_SIZE + blockDim.y - 1) / blockDim.y * COMPUTE_TILES; tile++){

            // read A matrix tile into shared memory
            int block_row = threadIdx.y + tile * blockDim.y;
            int row = global_row + block_row; // note the addition of tile * blockDim.y to read multiple tiles of A
            int col = k + threadIdx.x;

            T* ptr_sh_A = reinterpret_cast<T*>(&shared_A[block_row * (shared_mem_size_x) + threadIdx.x * TILE_SIZE]); // pointer to the tile in shared memory with padding
            T* ptr_A = reinterpret_cast<T*>(&A[row * K + col*TILE_SIZE]); // read 4 elements of A
            *ptr_sh_A = *ptr_A; // write to shared memory

            // read B matrix tile into shared memory, B is transposed
            row = global_col + block_row; // note the swap of row and col for B
            col = k + threadIdx.x;
            T* ptr_sh_B = reinterpret_cast<T*>(&shared_B[(block_row * (shared_mem_size_x) + threadIdx.x * TILE_SIZE)]); // pointer to the tile in shared memory with padding
            T* ptr_B = reinterpret_cast<T*>(&B[row * K + col*TILE_SIZE]); // read 4 elements of B
            *ptr_sh_B = *ptr_B; // write to shared memory

        }

        __syncthreads(); // ensure all threads have loaded their data into shared memory


        // compute C tile
        // load the tiles of A and B into WMMA fragments

        for(int tile = 0; tile < (blockDim.x * TILE_SIZE)/WMMA_SIZE; tile++){

            wmma::load_matrix_sync(a_frag, &shared_A[wmma_blockid_x*shared_mem_size_x + tile * WMMA_SIZE], shared_mem_size_x);
            wmma::load_matrix_sync(b_frag, &shared_B[wmma_blockid_y*shared_mem_size_x + tile * WMMA_SIZE], shared_mem_size_x);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        }
            // compute the tile of C using the loaded tiles of A and B
        __syncthreads(); // ensure all threads have completed their computation before the next iteration

    }

    int row = global_row + wmma_blockid_y * WMMA_SIZE;
    int col = global_col + wmma_blockid_x * WMMA_SIZE;

    if (row + 16 <= M && col + 16 <= N) 
        wmma::store_matrix_sync(&C[row * N + col], c_frag, N, 
            wmma::mem_row_major);

        // compute the tile of C using the loaded tiles of A and B
        // __half* ptr_a = reinterpret_cast<__half*>(&shared_A[0][0].x); // pointer to the first element of the tile in shared memory
        // __half* ptr_b = reinterpret_cast<__half*>(&shared_B[0][0].x); // pointer to the first element of the tile in shared memory
        // for(int t = 0; t < TILE_SIZE; t += 16){
        //     int col = threadIdx.x;
        //     wmma::load_matrix_sync(a_frag, &shared_A[threadIdx.y][t], size_tile_x);
        //     wmma::load_matrix_sync(b_frag, &shared_B[threadIdx.y][0], size_tile_y);
        //     wmma::fill_fragment(c_frag, 0.0f);
        //     wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        // }

        //  __syncthreads(); // ensure all threads have completed their computation before the next iteration

    

    // // compute the tile of C using the loaded tiles of A and B
    // int row = global_row + threadIdx.y;
    // int col = global_col + threadIdx.x;
    // C[row * N + col] = shared_A[threadIdx.y * 32 * TILE_SIZE + 0] * shared_B[0 * 32 * TILE_SIZE + threadIdx.x]; // initialize C with the first product
}

int main(){

    constexpr size_t M = 2*1024, N = 2*1024, K = 16*1024;

    constexpr size_t shared_bytes = 33 * 1024;
    // cudaFuncSetAttribute(
    //     gemm,
    //     cudaFuncAttributeMaxDynamicSharedMemorySize,
    //     shared_bytes
    // );
    cudaDeviceProp prop;
    cuda_check(cudaGetDeviceProperties(&prop, 0));


    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Maximum shared memory per block: " << prop.sharedMemPerBlock/1024 << " KB" << std::endl;
    std::cout << "Maximum shared memory per SM: " << prop.sharedMemPerMultiprocessor/1024 << " KB" << std::endl;
    std::cout << "Registers per block: " << prop.regsPerBlock << std::endl;
    std::cout << "warp size: " << prop.warpSize << std::endl;

    // Allocate gpu memory for A, B, C
    __half *d_A, *d_B;
    float *d_C;
    cuda_check(cudaMalloc(&d_A, M * K * sizeof(__half)));
    cuda_check(cudaMalloc(&d_B, K * N * sizeof(__half)));
    cuda_check(cudaMalloc(&d_C, M * N * sizeof(float)));

    dim3 blockDim(32, COMPUTE_TILES * COMPUTE_TILES); // 512 threads per block
    dim3 gridDim((N + WMMA_SIZE*COMPUTE_TILES - 1) / (WMMA_SIZE*COMPUTE_TILES),
         (M + WMMA_SIZE*COMPUTE_TILES - 1) / (WMMA_SIZE*COMPUTE_TILES)); // grid dimensions to cover the entire output matrix C
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