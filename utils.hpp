#include <iostream>

namespace Utils{

    template<typename T>
    static void Print_Vector(const T& vec, size_t rows, size_t cols){
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                std::cout << vec[i*cols + j] << " ";
            }
            std::cout << "\n";
        }
    }
}

// CUDA Error Checking
#define cuda_check(err) { \
    if (err != cudaSuccess) { \
        std::cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << "\n"; \
        exit(EXIT_FAILURE); \
    } \
}
