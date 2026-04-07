#include <type_traits>

#include <immintrin.h>
#include <type_traits>

template<typename T>
struct simd_traits {};

template<>
struct simd_traits<float> {
    using vec_type = __m256;
    static vec_type setzero() { return _mm256_setzero_ps(); }
    static vec_type set1(float val) { return _mm256_set1_ps(val); }
    static vec_type load(const float* ptr) { return _mm256_loadu_ps(ptr); }
    static vec_type fmadd(vec_type a, vec_type b, vec_type c) { return _mm256_fmadd_ps(a, b, c); }
    static void store(float* ptr, vec_type v) { _mm256_store_ps(ptr, v); }
    static float horizontal_sum(vec_type v) {
        alignas(32) float tmp[8];
        _mm256_store_ps(tmp, v);
        float sum = 0;
        for(int i=0;i<8;i++) sum += tmp[i];
        return sum;
    }
};

template<>
struct simd_traits<int> {
    using vec_type = __m256i;
    static vec_type setzero() { return _mm256_setzero_si256(); }
    static vec_type set1(int val) { return _mm256_set1_epi32(val); }
    static vec_type load(const int* ptr) { return _mm256_loadu_si256((__m256i*)ptr); }
    static vec_type mul(vec_type a, vec_type b) { return _mm256_mullo_epi32(a, b); }
    static vec_type add(vec_type a, vec_type b) { return _mm256_add_epi32(a, b); }
    static void store(int* ptr, vec_type v) { _mm256_store_si256((__m256i*)ptr, v); }
    static int horizontal_sum(vec_type v) {
        alignas(32) int tmp[8];
        _mm256_store_si256((__m256i*)tmp, v);
        int sum = 0;
        for(int i=0;i<8;i++) sum += tmp[i];
        return sum;
    }
};


namespace GEMM_CPU {

    template <typename T>
    requires (std::is_arithmetic_v<T>)
    void matmul_naive(T *A, T *B, T *C, int M, int N, int K) {

        for (int i = 0; i < M * N; ++i)
            C[i] = 0; // initialize C

        for (int i = 0; i < M; ++i) {
            for (int k = 0; k < K; k++) {
                T a = A[i*K+k];
                for (int j = 0; j < N; j++){
                    C[i * N + j] += a * B[k * N + j];
                }
            }
        }
    }

    template<typename T>
    requires (std::is_arithmetic_v<T>)
    void matmul_cache(T *A, T *B, T *C, int M, int N, int K ){
        constexpr int blockM = 64;
        constexpr int blockN = 64;
        constexpr int blockK = 64;

        for (int i = 0; i < M * N; ++i)
            C[i] = 0; // initialize C

        for(int i = 0; i < M; i+=blockM){
            for(int k = 0; k < K; k+= blockK){

                for(int j = 0; j < N; j+=blockN){

                    for(int bi = i; bi < i + blockM; bi++){
                        for(int bj = j; bj < j + blockN; bj++){
                            T c = C[bi*N+bj];
                            for(int bk = k; bk < k + blockK; bk++)
                                c += A[bi*K+bk] * B[bk*N+bj]; // avoid writes within cache friendly loops
                            C[bi*N+bj] = c;
                        }
                    }

                }


            }
        }

    }


    template<typename T>
    requires (std::is_arithmetic_v<T>)
    void matmul_cache_simd(T *A, T *B, T *C, int M, int N, int K ){
        constexpr int blockM = 64;
        constexpr int blockN = 64;
        constexpr int blockK = 64;

        using traits = simd_traits<T>;
        using vec_type = typename traits::vec_type;

        for(int i = 0; i < M; i+=blockM){
            for(int k = 0; k < K; k+= blockK){

                for(int j = 0; j < N; j+=blockN){

                    for(int bi = i; bi < i + blockM; bi++){
                        for(int bj = j; bj < j + blockN; bj++){
                            vec_type c_vec = traits::setzero();

                            for(int bk = k; bk < k + blockK; bk += 8) {
                                vec_type a_vec;
                                if constexpr(std::is_same_v<T,float>)
                                    a_vec = traits::set1(A[bi*K + bk]);
                                else
                                    a_vec = traits::set1(A[bi*K + bk]); // int version

                                vec_type b_vec = traits::load(&B[bk*N + bj]);

                                if constexpr(std::is_same_v<T,float>)
                                    c_vec = traits::fmadd(a_vec, b_vec, c_vec);
                                else
                                    c_vec = traits::add(c_vec, traits::mul(a_vec, b_vec));
                            }

                            C[bi*N + bj] += traits::horizontal_sum(c_vec);
                        }
                    }

                }


            }
        }

    }

}

