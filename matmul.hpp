
namespace GEMM_CPU {

    template <typename T>
    void matmul(T *A, T *B, T *C, int M, int N, int K) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                C[i * N + j] = 0;
                for (int k = 0; k < K; ++k) {
                    C[i * N + j] += A[i * K + k] * B[k * N + j];
                }
            }
        }
    }
    
}

