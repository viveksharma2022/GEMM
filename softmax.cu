#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cuda/pipeline>
#include "utils.hpp"
#include <cuda/barrier>
#include <vector>


// --- combine two (m, l) pairs ---
__device__ __forceinline__ void combine(float &m1, float &l1, float m2, float l2) {
    float m_new = fmaxf(m1, m2);
    float l_new = l1 * __expf(m1 - m_new) + l2 * __expf(m2 - m_new);
    m1 = m_new;
    l1 = l_new;
}

void __global__ softmax(vec4f* data, vec4f* output, size_t M, size_t N) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float m_local = -INFINITY;
    float l_local = 0.0f;

    for(int i = col ; i < N ; i+= blockDim.x * gridDim.x){

        float4 x = data[row * N + i];

        // process each element inside the float4
        float vals[4] = {x.x, x.y, x.z, x.w};

        #pragma unroll
        for(int j = 0; j < 4; j++){
            float x = vals[j];
            float new_m = fmaxf(m_local, x);
            float exp_m_diff = __expf(m_local - new_m);
            float exp_x_diff = __expf(x - new_m);

            float new_l = (l_local * exp_m_diff) + exp_x_diff;

            m_local = new_m;
            l_local = new_l;

        }

    }       

    // phase 2: warp reduction
    for(int offset = 16; offset > 0; offset /= 2){
        float m_other = __shfl_down_sync(0xffffffff, m_local, offset);
        float l_other = __shfl_down_sync(0xffffffff, l_local, offset);

        if(threadIdx.x < offset){
            combine(m_local, l_local, m_other, l_other);
        }
    }

    m_local = __shfl_sync(0xffffffff, m_local, 0); // broadcast final max to all threads in the block
    l_local = __shfl_sync(0xffffffff, l_local, 0);


    // write softmax output
        for(int i = col ; i < N ; i+= blockDim.x * gridDim.x){

                float4 x = data[row * N + i ];
                float4 m = make_float4(m_local, m_local, m_local, m_local); // all threads in the block have the same m and l
                float4 l = make_float4(l_local, l_local, l_local, l_local);
                float4 exp_x_diff = expf4(sub4(x,m));
                float4 softmax_val = div4(exp_x_diff, l);
                output[row * N + i] = softmax_val;
    }
}

int main() {

    size_t M = 576, N = 16*1024;
    vec4f* data;
    vec4f* output;
    dim3 blockDim(32, 1);
    dim3 gridDim(1,M);
    cuda_check(cudaMalloc(&data, M * N * sizeof(vec4f)));
    cuda_check(cudaMalloc(&output, M * N * sizeof(vec4f)));

    softmax<<<gridDim, blockDim>>>(data, output, M, N);
    cudaError_t err = cudaGetLastError();
    printf("Error: %s\n", cudaGetErrorString(err));
    cuda_check(cudaDeviceSynchronize());

    cuda_check(cudaFree(data));
    cuda_check(cudaFree(output));

    return 0;
}