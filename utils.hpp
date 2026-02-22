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

