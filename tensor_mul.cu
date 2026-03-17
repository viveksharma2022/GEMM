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

// constexpr int BLOCK_M = 128;
// constexpr int BLOCK_N = 128;

// constexpr int WARP_M = 32;
// constexpr int WARP_N = 32;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// constexpr int BLOCK_M = 64;
// constexpr int BLOCK_N = 64;

// constexpr int PAD = 8;

using half = __half;

// CUDA Error Checking
#define cuda_check(err) { \
    if (err != cudaSuccess) { \
        std::cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << "\n"; \
        exit(EXIT_FAILURE); \
    } \
}

template<typename A, typename B, typename C>
struct is_valid_tensor_combo {
    static constexpr bool value =
        std::is_same_v<A, __half> &&
        std::is_same_v<B, __half> &&
        std::is_same_v<C, float>;
};

template<typename MATRIXA, typename MATRIXB, typename MATRIXC>
__global__ void naive_tensor_mat_mul_kernel(
    MATRIXA* matA,
    MATRIXB* matB,
    MATRIXC* matC,
    int m,
    int n,
    int k)
{
    // tile using a 2D grid

    static_assert(is_valid_tensor_combo<MATRIXA, MATRIXB, MATRIXC>::value,
                  "Unsupported type combination");

    int tileM = blockIdx.x;
    int tileN = blockIdx.y;

    nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_a,
        WMMA_M,
        WMMA_N,
        WMMA_K,
        MATRIXA,
        nvcuda::wmma::row_major>
        a_frag;

    nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_b,
        WMMA_M,
        WMMA_N,
        WMMA_K,
        MATRIXB,
        nvcuda::wmma::row_major>
        b_frag;

    nvcuda::wmma::fragment<
        nvcuda::wmma::accumulator,
        WMMA_M,
        WMMA_N,
        WMMA_K,
        MATRIXC>
        c_frag;

    nvcuda::wmma::fill_fragment(c_frag, 0.0f);

    for (int i = 0; i < k; i += WMMA_K) {
        int aRow = tileM * WMMA_M;
        int aCol = i;
        int bRow = i;
        int bCol = tileN * WMMA_N;

        // Load the inputs
        nvcuda::wmma::load_matrix_sync(a_frag, &matA[aRow * k + aCol], k);
        nvcuda::wmma::load_matrix_sync(b_frag, &matB[bRow * n + bCol], n);

        // Perform the matrix multiplication
        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Store the output
    int cRow = tileM * WMMA_M;
    int cCol = tileN * WMMA_N;
    nvcuda::wmma::store_matrix_sync(
        &matC[cRow * n + cCol],
        c_frag,
        n,
        nvcuda::wmma::mem_row_major);
}

template<typename MATRIXA, typename MATRIXB, typename MATRIXC>
__global__ void shared_memory_tensor_mat_mul_kernel(MATRIXA* matA, MATRIXB* matB, MATRIXC* matC, int m, int n, int k){
    // Use shared memory to coalesce global memory reads
    // Pad rows so the shared-memory row stride stays aligned for WMMA loads.
    // For __half (2 bytes) WMMA prefers row stride in elements to be a multiple of 8
    // (so stride*2 bytes is divisible by 16). Use PAD = 8 to satisfy this.
    constexpr int PAD = 8;
    __shared__ MATRIXA sA[WMMA_M][WMMA_K + PAD];
    __shared__ MATRIXB sB[WMMA_K][WMMA_N + PAD];

    static_assert(is_valid_tensor_combo<MATRIXA,MATRIXB,MATRIXC>::value,
                  "Unsupported type combination");

    int tileM = blockIdx.x;
    int tileN = blockIdx.y;
    int tid = threadIdx.x;

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, MATRIXA, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, MATRIXB, nvcuda::wmma::row_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, MATRIXC> c_frag;

    nvcuda::wmma::fill_fragment(c_frag, 0.0f);

    int globalARow = tileM * WMMA_M;
    int globalBCol = tileN * WMMA_N;

    for(int i = 0; i < k; i+=WMMA_K){
        // Load matrix A tile: each thread loads one element
        if(tid < WMMA_M * WMMA_K){
            int row = tid / WMMA_K;
            int col = tid % WMMA_K;
            sA[row][col] = matA[(globalARow + row) * k + (i + col)];
        }

        // Load matrix B tile: each thread loads one element
        if(tid < WMMA_K * WMMA_N){
            int row = tid / WMMA_N;
            int col = tid % WMMA_N;
            sB[row][col] = matB[(i + row) * n + (globalBCol  + col)];
        }

        __syncthreads();

        // Load from shared memory into WMMA fragments using the padded leading dimensions
        nvcuda::wmma::load_matrix_sync(a_frag, &sA[0][0], WMMA_K + PAD);
        nvcuda::wmma::load_matrix_sync(b_frag, &sB[0][0], WMMA_N + PAD);

        __syncthreads();

        // Perform the matrix multiplication
        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    }

    // Store the output
    int cRow = tileM * WMMA_M;
    int cCol = tileN * WMMA_N;
    nvcuda::wmma::store_matrix_sync(&matC[cRow * n + cCol], c_frag, n, nvcuda::wmma::mem_row_major);
    
}

template<typename MATRIXA, typename MATRIXB, typename MATRIXC, size_t N_Tiles = 1>
__global__ void shared_memory_increase_tensor_mat_mul_kernel(MATRIXA* matA, MATRIXB* matB, MATRIXC* matC, int m, int n, int k){
    // Use shared memory to coalesce global memory reads
    // Pad rows so the shared-memory row stride stays aligned for WMMA loads.
    // For __half (2 bytes) WMMA prefers row stride in elements to be a multiple of 8
    // (so stride*2 bytes is divisible by 16). Use PAD = 8 to satisfy this.
    constexpr int PAD = 8;
    __shared__ MATRIXA sA[WMMA_M ][WMMA_K * N_Tiles + PAD];
    __shared__ MATRIXB sB[WMMA_K ][WMMA_N * N_Tiles + PAD];
    
    static_assert(is_valid_tensor_combo<MATRIXA,MATRIXB,MATRIXC>::value,
                  "Unsupported type combination");

    int tileM = blockIdx.x;
    int tileN = blockIdx.y;
    int tid = threadIdx.x;



    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, MATRIXA, nvcuda::wmma::row_major> a_frag[N_Tiles];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, MATRIXB, nvcuda::wmma::row_major> b_frag[N_Tiles];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, MATRIXC> c_frag[N_Tiles];

    for(int i = 0; i < N_Tiles; i++){
        nvcuda::wmma::fill_fragment(c_frag[i], 0.0f);
    }

    int globalARow = tileM * WMMA_M;
    int globalBCol = tileN * WMMA_N;

    for(int i = 0; i < k; i+=WMMA_K){
        // Load matrix A tile: each thread loads one element
        if(tid < WMMA_M * WMMA_K){
            int row = tid / WMMA_K;
            int col = tid % WMMA_K;
            sA[row][col] = matA[(globalARow + row) * k + (i + col)];
        }

        // Load matrix B tile: each thread loads one element
        if(tid < WMMA_K * WMMA_N){
            int row = tid / WMMA_N;
            int col = tid % WMMA_N;
            sB[row][col] = matB[(i + row) * n + (globalBCol  + col)];
        }

        __syncthreads();


        // Load from shared memory into WMMA fragments using the padded leading dimensions
        for(int t = 0; t < N_Tiles; t++){
            nvcuda::wmma::load_matrix_sync(a_frag[t], &sA[0][t*(WMMA_K)], WMMA_K*N_Tiles + PAD);
            nvcuda::wmma::load_matrix_sync(b_frag[t], &sB[0][t*(WMMA_N)], WMMA_N*N_Tiles + PAD);
        }

        __syncthreads();    

        // Perform the matrix multiplication
        for(int t = 0; t < N_Tiles; t++)
            nvcuda::wmma::mma_sync(c_frag[t], a_frag[t], b_frag[t], c_frag[t]);

    }

    // Store the output
    for(int t = 0; t < N_Tiles; t++){
        int cRow = tileM * WMMA_M;
        int cCol = tileN * WMMA_N + t*WMMA_N;   
        nvcuda::wmma::store_matrix_sync(&matC[cRow * n + cCol], c_frag[t], n, nvcuda::wmma::mem_row_major);
    }
}


template<typename TA, typename TB, typename TC,
size_t NTiles = 1, size_t Block_M = 64, size_t Block_N = 64, 
int WMMA_M = 16, int WMMA_N = 16, int WMMA_K = 16>
__global__ void lds_load(
    const TA* __restrict__ A,
    const TB* __restrict__ B,
    TC* __restrict__ C,
    int M, int N, int K
){

    constexpr size_t Block_K = 16;
    constexpr size_t PAD = 8; 

    __shared__ alignas(16) TA sA[Block_M][Block_K + PAD];
    __shared__ alignas(16) TB sB[Block_K][Block_N + PAD];

    int tid = threadIdx.x;

    int blockRow = blockIdx.y * Block_M;
    int blockCol = blockIdx.x * Block_N;

    #pragma unroll
        for(int r = 0; r < NTiles/2; r++){
            for(int c = 0; c < NTiles/2; c++){
                // load tile A: along rows
                int row = tid / Block_K;
                int col = tid % Block_K;
                int global_row = blockRow + r * (Block_M / (NTiles/2)) + row;
                int global_col = c * Block_K + col;

                if(global_row < M && global_col < K)
                    sA[row][col] = A[global_row * K + global_col];
                else
                    sA[row][col] = 0;

                // load tile B: along columns
                row = tid / Block_N;
                col = tid % Block_N;
                global_row = r * Block_K + row;
                global_col = blockCol + c * (Block_N / (NTiles/2)) + col;

                if(global_row < K && global_col < N)
                    sB[row][col] = B[global_row * N + global_col];
                else
                    sB[row][col] = 0;
            }
        }

    __syncthreads();
    float val = 0.0f;

    for(int i=0;i<Block_K;i++)
        val += __half2float(sA[threadIdx.x % Block_M][i]) *
            __half2float(sB[i][threadIdx.x % Block_N]);

    
    if (blockRow < M && blockCol < N)
        C[blockRow * N + blockCol] = val;
    
}


template<
    typename TA,
    typename TB,
    typename TC,
    int BLOCK_M = 64,
    int BLOCK_N = 64,
    int BLOCK_K = 64>
__global__ void lds_load_vec(
    const TA* __restrict__ A,
    const TB* __restrict__ B,
    TC* __restrict__ C,
    int M, int N, int K)
{
    constexpr int PAD = 8;

    // int2 = 8 bytes = 4 half
    constexpr int VEC = 8;  // number of half elements per int2 load

    using DT = int2;

    // Shared tiles with padding to reduce bank conflicts
    __shared__ __shared__ __align__(128) TA sA[BLOCK_M][BLOCK_K + PAD];
    __shared__ __shared__ __align__(128) TB sB[BLOCK_K][BLOCK_N + PAD];

    int tid = threadIdx.x;

    // Output tile coordinates
    int blockRow = blockIdx.y * BLOCK_M;
    int blockCol = blockIdx.x * BLOCK_N;

    //--------------------------------
    // vectorized global memory views
    //--------------------------------

    const DT* A_vec = reinterpret_cast<const DT*>(A);
    const DT* B_vec = reinterpret_cast<const DT*>(B);

    //--------------------------------
    // tile dimensions in vector units
    //--------------------------------

    // number of int2 loads per row 
    constexpr int A_VEC_WIDTH = BLOCK_K / VEC;  // 64 / 8 = 8
    constexpr int B_VEC_WIDTH = BLOCK_N / VEC;  // 64 / 8 = 8

    // total vector loads required to fill each tile
    constexpr int A_VEC_COUNT = BLOCK_M * A_VEC_WIDTH; // 64 * 8 = 512
    constexpr int B_VEC_COUNT = BLOCK_K * B_VEC_WIDTH; // 64 * 8 = 512

    for(int k = 0; k < K; k += BLOCK_K){
    //--------------------------------
        // load A tile (BLOCK_M x BLOCK_K)
        //--------------------------------

        #pragma unroll
        for (int i = tid; i < A_VEC_COUNT; i += blockDim.x)
        {
            int row = i / A_VEC_WIDTH;
            int col = i % A_VEC_WIDTH;

            // global index in int2 units
            int gmem =
                (blockRow + row) * (K / VEC) + col + (k / VEC);

            DT* sA_vec = reinterpret_cast<DT*>(&sA[row][col*VEC]);
            *sA_vec = A_vec[gmem];
        }

        //--------------------------------
        // load B tile (BLOCK_K x BLOCK_N)
        //--------------------------------

        #pragma unroll
        for (int i = tid; i < B_VEC_COUNT; i += blockDim.x)
        {
            int row = i / B_VEC_WIDTH;
            int col = i % B_VEC_WIDTH;

            // global index in int2 units
            int gmem =
                (k + row) * (N / VEC) + (blockCol / VEC) + col;
                

            DT* sB_vec = reinterpret_cast<DT*>(&sB[row][col*VEC]);
            *sB_vec = B_vec[gmem];
        }

        __syncthreads();

            // WMMA compute: 16 tiles per block
        int tile = 0;
        for (int i = 0; i < BLOCK_M; i += WMMA_M) {       // 4 rows
            for (int j = 0; j < BLOCK_N; j += WMMA_N) {   // 4 cols
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

                wmma::fill_fragment(c_frag, 0.0f);

                for (int t = 0; t < BLOCK_K; t += WMMA_K) {
                    wmma::load_matrix_sync(a_frag, &sA[i][t], BLOCK_K + PAD);
                    wmma::load_matrix_sync(b_frag, &sB[t][j], BLOCK_N + PAD);
                    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                }

                wmma::store_matrix_sync(&C[(blockRow + i) * N + (blockCol + j)], c_frag, N, wmma::mem_row_major);
                tile++;
            }
        }

        __syncthreads();

    }
 
    //--------------------------------
    // placeholder compute
    //--------------------------------

    //--------------------------------
    // simple coalesced store
    //--------------------------------

    // #pragma unroll
    // for (int i = tid; i < BLOCK_M * BLOCK_N; i += blockDim.x)
    // {
    //     int row = i / BLOCK_N;
    //     int col = i % BLOCK_N;

    //     int global_row = blockRow + row;
    //     int global_col = blockCol + col;

    //     float val = 0.0f;

    //     for (int k = 0; k < BLOCK_K; k++)
    //     {
    //         val += __half2float(sA[row][k]) *
    //             __half2float(sB[k][col]);
    //     }

    //     if (global_row < M && global_col < N)
    //     {
    //         C[global_row * N + global_col] = val;
    //     }
    // }
}

int main(){


    int device = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    std::cout << "Max shared memory per block: " 
              << prop.sharedMemPerBlock / 1024.0f 
              << " KB\n";

    std::cout << "Max shared memory per multiprocessor: "
              << prop.sharedMemPerMultiprocessor / 1024.0f
              << " KB\n";

    std::cout << "Compute capability: "
              << prop.major << "." << prop.minor << "\n";
    
    using type_A = __half;
    using type_B = __half;
    using type_C = float;

    int M = 2 * 1024, N = 2 * 1024, K = 32 * 1024;

    // constexpr size_t N_TILES = 4; // Number of tiles to compute per block (for shared_memory_increase_tensor_mat_mul_kernel)

    std::vector<type_A> A(M*K, 1.0f);
    std::vector<type_B> B(K*N, 1.0f);
    std::vector<type_C> C(M*N, 0.0f);

    float *d_C;
    half *d_A, *d_B;
    cuda_check(cudaMalloc((void**)&d_A, A.size()* sizeof(std::remove_reference_t<decltype(A[0])>)));
    cuda_check(cudaMalloc((void**)&d_B, B.size()* sizeof(std::remove_reference_t<decltype(B[0])>)));
    cuda_check(cudaMalloc((void**)&d_C, C.size()* sizeof(std::remove_reference_t<decltype(C[0])>)));

    cuda_check(cudaMemcpy(d_A, A.data(),
                          A.size() * sizeof(type_A),
                          cudaMemcpyHostToDevice));

    cuda_check(cudaMemcpy(d_B, B.data(),
                          B.size() * sizeof(type_B),
                          cudaMemcpyHostToDevice));

    constexpr int BLOCK_M = 64;  // 4 WMMAs tall (4*16)
    constexpr int BLOCK_N = 64;  // 4 WMMAs wide (4*16)
    constexpr int BLOCK_K = 64;  // 4 WMMAs deep (4*16)
    constexpr int threadBlockSize = 256;  // 16 warps, each thread loads 64 bytes

    dim3 dim_block(threadBlockSize);
    dim3 dim_grid(N / BLOCK_N, M / BLOCK_M);

    printf("Launching kernel with grid (%d, %d) and block (%d, %d)\n",
           dim_grid.x, dim_grid.y, dim_block.x, dim_block.y);
    printf("Per block: 64 WMMAs (8×8 WMMA grid; 8 warps each compute 8 WMMA tiles sequentially)\n");
    
        lds_load_vec<half, half, float, BLOCK_M, BLOCK_N, BLOCK_K>
        <<<dim_grid, dim_block>>>(d_A, d_B, d_C, M, N, K);

    // // Call optimized kernel with shared memory
    // printf("Executing tensor multiplication naive\n");
    // naive_tensor_mat_mul_kernel<<<dim_grid, dim_block>>>(d_A, d_B, d_C, M, N, K);

    // printf("Executing tensor multiplication with shared memory optimization\n");
    // shared_memory_tensor_mat_mul_kernel<<<dim_grid, dim_block>>>(d_A, d_B, d_C, M, N, K);

    // dim3 dim_block1(256 * N_TILES, 1);
    // dim3 dim_grid1(M/(WMMA_M * N_TILES), N/WMMA_N);

    // printf("Executing tensor multiplication with shared memory optimization and increased load\n");
    // shared_memory_increase_tensor_mat_mul_kernel<type_A, type_B, type_C, N_TILES><<<dim_grid1, dim_block1>>>(d_A, d_B, d_C, M, N, K);

    // dim3 dim_block2(512, 1);
    // dim3 dim_grid2(M/128, N/128);

    // printf("Executing tensor multiplication with shared memory optimization and double buffering\n");
    // wmma_gemm_64x64_4mma<<<dim_grid, dim_block>>>(d_A, d_B, d_C, M, N, K);

    // printf("Executing tensor multiplication with shared memory optimization and simple\n");
    // wmma_gemm_64x64_simple<<<dim_grid, dim_block>>>(d_A, d_B, d_C, M, N, K);

    // printf("Executing tensor multiplication with shared memory optimization and vector load\n");
    // wmma_gemm_64x64_vec_async<<<dim_grid, dim_block>>>(d_A, d_B, d_C, M, N, K);

    cuda_check(cudaDeviceSynchronize());

    cuda_check(cudaMemcpy(C.data(), d_C,
                          C.size() * sizeof(type_C),
                          cudaMemcpyDeviceToHost));

    // Print result
    std::cout << "Result matrix C:\n";
    // Utils::Print_Vector(C, M, N);

    cuda_check(cudaFree(d_A));
    cuda_check(cudaFree(d_B));
    cuda_check(cudaFree(d_C));
    return 0;
}