#include "matmul.hpp"
#include <memory>

int main(){

    size_t M = 512, N = 512, K = 512;
    using T = float;

    auto A = std::make_unique<T[]>(M * K);
    auto B = std::make_unique<T[]>(K * N);   
    auto C = std::make_unique<T[]>(M * N);

    GEMM_CPU::matmul(A.get(), B.get(), C.get(), M, N, K);

    return 0;
}