#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <xmmintrin.h>
#include <bit>

constexpr int head_dim = 64;

template<typename T, size_t N>
__forceinline T vector_sum(const T* a, const T* b) {
 
    constexpr int simd_width = 128/sizeof(T); // __m128 can hold 4 floats
    __m128 sum_vec = _mm_setzero_ps();
                
    int j = 0;
    for(; j + simd_width <=N; j+=simd_width){
        __m128 a_vec = _mm_loadu_ps(a + j);
        __m128 b_vec = _mm_loadu_ps(b + j);
        sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(a_vec, b_vec));
    }

    // horizontal sum
    sum_vec = _mm_add_ps(sum_vec, _mm_movehl_ps(sum_vec, sum_vec));
    sum_vec = _mm_add_ss(sum_vec, _mm_shuffle_ps(sum_vec, sum_vec, 1));
    float sum = _mm_cvtss_f32(sum_vec);

    // remainder
    for (; j < N; j++) {
        sum += a[j] * b[j];
    }
    return sum;
}

// linear projection to get Q, K, V
void linear(const  float* __restrict x, const float* __restrict W, float* y) {

    for (int i = 0; i < head_dim; i++) {

        const float* w_row = W + i * head_dim;
        y[i] = vector_sum<float, head_dim>(x, w_row);
    }
}

// compute q, k, v from input token x_t
void compute_qkv(const  float* __restrict x_t, const  float* __restrict W_q, const  float* __restrict W_k, const  float* __restrict W_v,
                 float* q_t, float* k_t, float* v_t) {

    linear(x_t, W_q, q_t);
    linear(x_t, W_k, k_t);
    linear(x_t, W_v, v_t);
}

// flash attention (CPU) with KV cache
void flash_attention_cpu_kv(
    const  float* __restrict q,
    const  float* __restrict K_cache,
    const  float* __restrict V_cache,
    float* out,
    int t
){

    float m = - INFINITY;
    float l = 0.0f;
    float acc[head_dim] = {0};

    for(int i = 0; i < t; i++){

        const float* k = K_cache + i * head_dim;
        const float* v = V_cache + i * head_dim;

        float score = vector_sum<float, head_dim>(q, k) / std::sqrt((float)head_dim);

        float new_m = std::max(m, score);
        float exp_old = std::exp(m - new_m) * l;
        float exp_new = std::exp(score - new_m);
        float new_l = exp_old + exp_new;

        float scale_old = (l == 0.0f) ? 0.0f : (exp_old / new_l);
        float scale_new = exp_new / new_l;

        for(int d = 0; d < head_dim; d++){
            acc[d] = acc[d] * scale_old + (scale_new * v[d]);
        }

         m = new_m;
         l = new_l;
    }

    std::memcpy(out, acc, head_dim * sizeof(float));

}   


int main(){

    const int seq_len = 512;

    std::vector<float> K_cache(seq_len * head_dim);
    std::vector<float> V_cache(seq_len * head_dim);

    std::vector<float> W_q(head_dim * head_dim);
    std::vector<float> W_k(head_dim * head_dim);    
    std::vector<float> W_v(head_dim * head_dim);

    //initialize weights dummy but consistent
    for (int i = 0; i < head_dim * head_dim; i++) {
        W_q[i] = (float)rand() / RAND_MAX;
        W_k[i] = (float)rand() / RAND_MAX;
        W_v[i] = (float)rand() / RAND_MAX;
    }

    std::vector<float> x(head_dim);
    std::vector<float> q(head_dim), k(head_dim), v(head_dim);
    std::vector<float> out(head_dim);


    for(int t = 0; t < seq_len; t++){

        //input token embedding dummy
        for (int d = 0; d < head_dim; d++){
            x[d] = static_cast<float>(rand()/RAND_MAX);
        }

        compute_qkv(x.data(), W_q.data(), W_k.data(), W_v.data(),
                    q.data(), k.data(), v.data());

        // update kv cache
        std::memcpy(K_cache.data() + t * head_dim, k.data(), head_dim * sizeof(float));
        std::memcpy(V_cache.data() + t * head_dim, v.data(), head_dim * sizeof(float));
            
        //run attention using cache
        flash_attention_cpu_kv(
            q.data(),
            K_cache.data(),
            V_cache.data(),
            out.data(),
            t + 1
        );

    }

}