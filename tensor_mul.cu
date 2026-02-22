#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>
#include <cuda_fp16.h>
#include <vector>
#include <iostream>
#include <type_traits>

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

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
__global__ void naive_tensor_mat_mul_kernel(MATRIXA* matA, MATRIXB* matB, MATRIXC* matC, int m, int n, int k){

    // tile using a 2D grid

    static_assert(is_valid_tensor_combo<MATRIXA,MATRIXB,MATRIXC>::value,
                  "Unsupported type combination");

    int warpM = blockIdx.x;
    int warpN = blockIdx.y;

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, MATRIXA, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, MATRIXB, nvcuda::wmma::row_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, MATRIXC> c_frag;

    nvcuda::wmma::fill_fragment(c_frag, 0.0f);

    for(int i = 0; i < k; i+=WMMA_K){
        int aRow = warpM * WMMA_M;
        int aCol = i;
        int bRow = i;
        int bCol = warpN * WMMA_N;

        // Load the inputs
        nvcuda::wmma::load_matrix_sync(a_frag, &matA[aRow * k + aCol], k);
        nvcuda::wmma::load_matrix_sync(b_frag, &matB[bRow * n + bCol], n);

        // Perform the matrix multiplication
        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

        // Store the output
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;
    nvcuda::wmma::store_matrix_sync(&matC[cRow * n + cCol], c_frag, n, nvcuda::wmma::mem_row_major);
}

int main(){

    using type_A = __half;
    using type_B = __half;
    using type_C = float;

    int M = WMMA_M, N = WMMA_N, K = WMMA_K;

    printf("Executing tensor multiplication naive");

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

    dim3 dim_block(32, 1);
    dim3 dim_grid(M/WMMA_M, N/WMMA_N);
    naive_tensor_mat_mul_kernel<<<dim_grid, dim_block>>>(d_A, d_B, d_C, M, N, K);

    cuda_check(cudaDeviceSynchronize());

    cuda_check(cudaMemcpy(C.data(), d_C,
                          C.size() * sizeof(type_C),
                          cudaMemcpyDeviceToHost));

    // Print result
    std::cout << "Result matrix C:\n";
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << C[i*N + j] << " ";
        }
        std::cout << "\n";
    }

    cuda_check(cudaFree(d_A));
    cuda_check(cudaFree(d_B));
    cuda_check(cudaFree(d_C));
    return 0;
}