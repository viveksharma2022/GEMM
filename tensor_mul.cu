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


#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;

template <
    typename TA,
    typename TB,
    typename TC,
    int BLOCK_M = 64,
    int BLOCK_N = 64,
    int BLOCK_K = 64>
__global__ void wmma_safe_kernel(
    const TA* __restrict__ A,
    const TB* __restrict__ B,
    TC* __restrict__ C,
    int M, int N, int K)
{
    constexpr int PAD = 8;                  // padding to reduce bank conflicts
    constexpr int VEC = 8;                  // half elements per int2 load
    using DT = int4;                        // vectorized type for global memory

    // WMMA tile size
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;

    // Shared memory tiles
    __shared__ __align__(16) TA sA[BLOCK_M][BLOCK_K + PAD];
    __shared__ __align__(16) TB sB[BLOCK_K][BLOCK_N + PAD];

    int tid = threadIdx.x;
    int warpId = tid / 32;
    int laneId = tid % 32;

    // Block coordinates in global matrix
    int blockRow = blockIdx.y * BLOCK_M;
    int blockCol = blockIdx.x * BLOCK_N;

    // Warp layout inside block
    constexpr int WARPS_PER_ROW = BLOCK_N / WMMA_N;  // 64/16 = 4
    int warpRow = warpId / WARPS_PER_ROW;
    int warpCol = warpId % WARPS_PER_ROW;

    int tileRow = warpRow * WMMA_M; // starting row of this warp's tile
    int tileCol = warpCol * WMMA_N; // starting col of this warp's tile

    // Vectorized global memory views
    const DT* A_vec = reinterpret_cast<const DT*>(A);
    const DT* B_vec = reinterpret_cast<const DT*>(B);

    constexpr int A_VEC_WIDTH = BLOCK_K / VEC;
    constexpr int B_VEC_WIDTH = BLOCK_N / VEC;
    constexpr int A_VEC_COUNT = BLOCK_M * A_VEC_WIDTH;
    constexpr int B_VEC_COUNT = BLOCK_K * B_VEC_WIDTH;

    // Accumulator fragment
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, TC> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    // Loop over K dimension in BLOCK_K chunks
    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {

        // ------------------------
        // Load A tile into shared memory (vectorized)
        // ------------------------
        for (int i = tid; i < A_VEC_COUNT; i += blockDim.x) {
            int row = i / A_VEC_WIDTH;
            int col = i % A_VEC_WIDTH;

            int globalRow = blockRow + row;
            int globalCol = k0 + col * VEC;

            if (globalRow < M && globalCol + VEC <= K) {
                DT* sA_vec = reinterpret_cast<DT*>(&sA[row][col * VEC]);
                *sA_vec = A_vec[(globalRow * (K / VEC)) + (globalCol / VEC)];
            }
        }

        // ------------------------
        // Load B tile into shared memory (vectorized)
        // ------------------------
        for (int i = tid; i < B_VEC_COUNT; i += blockDim.x) {
            int row = i / B_VEC_WIDTH;
            int col = i % B_VEC_WIDTH;

            int globalRow = k0 + row;
            int globalCol = blockCol + col * VEC;

            if (globalRow + VEC <= K && globalCol < N) {
                DT* sB_vec = reinterpret_cast<DT*>(&sB[row][col * VEC]);
                *sB_vec = B_vec[(globalRow * (N / VEC)) + (globalCol / VEC)];
            }
        }

        __syncthreads();

        // ------------------------
        // Compute WMMA fragments per warp
        // ------------------------
        for (int kk = 0; kk < BLOCK_K; kk += WMMA_K) {

            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, TA, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, TB, wmma::row_major> b_frag;

            // Load fragment from shared memory
            wmma::load_matrix_sync(a_frag, &sA[tileRow][kk], BLOCK_K + PAD);
            wmma::load_matrix_sync(b_frag, &sB[kk][tileCol], BLOCK_N + PAD);

            // Perform matrix multiply-accumulate
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        __syncthreads();
    }

    // ------------------------
    // Store result fragment
    // ------------------------
    wmma::store_matrix_sync(&C[(blockRow + tileRow) * N + (blockCol + tileCol)],
                            c_frag, N, wmma::mem_row_major);
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
// }

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
    
    using TA = __half;
    using TB = __half;
    using TC = __half;

    int M = 1 * 1024, N = 2 * 1024, K = 32 * 1024;

    // constexpr size_t N_TILES = 4; // Number of tiles to compute per block (for shared_memory_increase_tensor_mat_mul_kernel)

    std::vector<TA> A(M*K, 1.0f);
    std::vector<TB> B(K*N, 1.0f);
    std::vector<TC> C(M*N, 0.0f);

    TC *d_C;
    TA *d_A;
    TB *d_B;
    cuda_check(cudaMalloc((void**)&d_A, A.size()* sizeof(std::remove_reference_t<decltype(A[0])>)));
    cuda_check(cudaMalloc((void**)&d_B, B.size()* sizeof(std::remove_reference_t<decltype(B[0])>)));
    cuda_check(cudaMalloc((void**)&d_C, C.size()* sizeof(std::remove_reference_t<decltype(C[0])>)));

    cuda_check(cudaMemcpy(d_A, A.data(),
                          A.size() * sizeof(TA),
                          cudaMemcpyHostToDevice));

    cuda_check(cudaMemcpy(d_B, B.data(),
                          B.size() * sizeof(TB),
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
    
        wmma_safe_kernel<TA, TB, TC, BLOCK_M, BLOCK_N, BLOCK_K>
        <<<dim_grid, dim_block>>>(d_A, d_B, d_C, M, N, K);

    // // Call optimized kernel with shared memory
    // printf("Executing tensor multiplication naive\n");
    // naive_tensor_mat_mul_kernel<<<dim_grid, dim_block>>>(d_A, d_B, d_C, M, N, K);

    // printf("Executing tensor multiplication with shared memory optimization\n");
    // shared_memory_tensor_mat_mul_kernel<<<dim_grid, dim_block>>>(d_A, d_B, d_C, M, N, K);

    // dim3 dim_block1(256 * N_TILES, 1);
    // dim3 dim_grid1(M/(WMMA_M * N_TILES), N/WMMA_N);

    // printf("Executing tensor multiplication with shared memory optimization and increased load\n");
    // shared_memory_increase_tensor_mat_mul_kernel<TC, type_B, type_C, N_TILES><<<dim_grid1, dim_block1>>>(d_A, d_B, d_C, M, N, K);

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
                          C.size() * sizeof(TC),
                          cudaMemcpyDeviceToHost));

    // Print result
    std::cout << "Result matrix C:\n";
    // Utils::Print_Vector(C, M, N);

    cuda_check(cudaFree(d_A));
    cuda_check(cudaFree(d_B));
    cuda_check(cudaFree(d_C));
    return 0;


    
}

// #include <iostream>
// #include <cuda_runtime.h>

// // CUTLASS GEMM API
// #include "cutlass/cutlass.h"
// #include "cutlass/gemm/device/gemm.h"

// #define CUDA_CHECK(ans) { cudaAssert((ans), __FILE__, __LINE__); }
// inline void cudaAssert(cudaError_t code, const char *file, int line)
// {
//     if (code != cudaSuccess) {
//         std::cerr << "CUDA Error: " << cudaGetErrorString(code)
//                   << " at " << file << ":" << line << "\n";
//         exit(EXIT_FAILURE);
//     }
// }

// int main() {
//     // Matrix sizes
//     int M = 1024, N = 2*1024, K =32*1024;

//     // Host data
//     std::vector<cutlass::half_t> A(M * K, cutlass::half_t(1.0f));
//     std::vector<cutlass::half_t> B(K * N, cutlass::half_t(1.0f));
//     std::vector<cutlass::half_t> C(M * N, cutlass::half_t(0.0f));

//     // Device buffers
//     cutlass::half_t* d_A;
//     cutlass::half_t* d_B;
//     cutlass::half_t* d_C;

//     CUDA_CHECK(cudaMalloc(&d_A, sizeof(cutlass::half_t) * M * K));
//     CUDA_CHECK(cudaMalloc(&d_B, sizeof(cutlass::half_t) * K * N));
//     CUDA_CHECK(cudaMalloc(&d_C, sizeof(cutlass::half_t) * M * N));

//     CUDA_CHECK(cudaMemcpy(d_A, A.data(), sizeof(cutlass::half_t) * M * K, cudaMemcpyHostToDevice));
//     CUDA_CHECK(cudaMemcpy(d_B, B.data(), sizeof(cutlass::half_t) * K * N, cudaMemcpyHostToDevice));

//     //
//     // Define CUTLASS GEMM type
//     //
//     using Gemm = cutlass::gemm::device::Gemm<
//         cutlass::half_t,                // element A
//         cutlass::layout::RowMajor,      // layout A
//         cutlass::half_t,                // element B
//         cutlass::layout::RowMajor,      // layout B
//         cutlass::half_t,                // element C / D (output)
//         cutlass::layout::RowMajor,
//         float,                          // accumulator type
//         cutlass::arch::OpClassTensorOp, // target Tensor Cores
//         cutlass::arch::Sm80,            // compute capability (adjust as needed)
//         cutlass::gemm::GemmShape<64,64,64>,     // Threadblock tile shape
//         cutlass::gemm::GemmShape<32,32,32>,       // Warp tile shape
//         cutlass::gemm::GemmShape<16,8,16>,        // Instruction tile
//         cutlass::epilogue::thread::LinearCombination<
//             cutlass::half_t, 1, float, float>,
//         cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
//         2 /* Stages */
//     >;

//     Gemm gemm_op;

//     cutlass::Status status;

//     // Arguments: {ProblemSize}, {A, lda}, {B, ldb}, {C, ldc}, {D, ldd}, {alpha, beta}
//     status = gemm_op({
//         { M, N, K },
//         { d_A, K },
//         { d_B, N },
//         { d_C, N },
//         { d_C, N },
//         { 1.0f, 0.0f }
//     });

//     if (status != cutlass::Status::kSuccess) {
//         std::cerr << "CUTLASS GEMM failed with status: " << int(status) << "\n";
//         return -1;
//     }

//     CUDA_CHECK(cudaMemcpy(C.data(), d_C, sizeof(cutlass::half_t) * M * N, cudaMemcpyDeviceToHost));

//     std::cout << "C[0] = " << float(C[0]) << "\n";

//     cudaFree(d_A);
//     cudaFree(d_B);
//     cudaFree(d_C);

//     return 0;
// }