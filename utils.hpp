#include <cuda_runtime.h>

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

using vec4f = float4;

__inline__ __device__ float4 add4(vec4f a, vec4f b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__inline__ __device__ float4 sub4(vec4f a, vec4f b) {
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

__inline__ __device__ float4 fma4(float a, vec4f b, vec4f c) {
    return make_float4(fmaf(a, b.x, c.x), fmaf(a, b.y, c.y), fmaf(a, b.z, c.z), fmaf(a, b.w, c.w));
}

__device__ float4 fmaxf4(float4 a, float4 b) {
    return make_float4(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z), fmaxf(a.w, b.w));
}

__device__ float4 expf4(float4 x) {
    return make_float4(__expf(x.x), __expf(x.y), __expf(x.z), __expf(x.w));
}

__device__ float4 div4(float4 a, float4 b) {
    return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}

__device__ float4 mul4(float4 a, float4 b) {
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

__inline__ __device__
float4 warpReduceFMA(float4 val, float scale) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val.x = fmaf(scale, __shfl_down_sync(0xffffffff, val.x, offset), val.x);
        val.y = fmaf(scale, __shfl_down_sync(0xffffffff, val.y, offset), val.y);
        val.z = fmaf(scale, __shfl_down_sync(0xffffffff, val.z, offset), val.z);
        val.w = fmaf(scale, __shfl_down_sync(0xffffffff, val.w, offset), val.w);
    }
    return val;
}