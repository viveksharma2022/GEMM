#include "matmul.hpp"
#include <memory>
#include <iostream>
#include <chrono>
#include <cassert>      // for assert

int main(){

    size_t M = 512, N = 512, K = 512;
    constexpr int alignment = 64;
    using T = int;

    auto deleter = [](T* ptr) {
        _aligned_free(ptr); // must use _aligned_free
    };

    std::unique_ptr<T[], decltype(deleter)> A(
        static_cast<T*>(_aligned_malloc(M*K*sizeof(T), alignment)),
        deleter
    );

    std::unique_ptr<T[], decltype(deleter)> B(
        static_cast<T*>(_aligned_malloc(K*N*sizeof(T), alignment)),
        deleter
    );

    std::unique_ptr<T[], decltype(deleter)> C1(
        static_cast<T*>(_aligned_malloc(M*N*sizeof(T), alignment)),
        deleter
    );

    std::unique_ptr<T[], decltype(deleter)> C2(
        static_cast<T*>(_aligned_malloc(M*N*sizeof(T), alignment)),
        deleter
    );

    std::unique_ptr<T[], decltype(deleter)> C3(
        static_cast<T*>(_aligned_malloc(M*N*sizeof(T), alignment)),
        deleter
    );


    auto start = std::chrono::high_resolution_clock::now();
    GEMM_CPU::matmul_naive(A.get(), B.get(), C1.get(), M, N, K);
    auto end = std::chrono::high_resolution_clock::now();
    auto time_ns = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    auto start1 = std::chrono::high_resolution_clock::now();
    GEMM_CPU::matmul_cache(A.get(), B.get(), C2.get(), M, N, K);
    auto end1 = std::chrono::high_resolution_clock::now();
    auto time_ns1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1);

    auto start2 = std::chrono::high_resolution_clock::now();
    GEMM_CPU::matmul_cache_simd(A.get(), B.get(), C3.get(), M, N, K);
    auto end2 = std::chrono::high_resolution_clock::now();
    auto time_ns2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2);

    std::cout << "Time taken 1: " << time_ns << "\n";
    std::cout << "Time taken 2: " << time_ns1 << "\n";
    std::cout << "Time taken 2: " << time_ns2 << "\n";

    // validation
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            assert(C1[i*N+j] == C3[i*N+j]);
        }
    }

    return 0;
}