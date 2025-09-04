#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <half>

#define MIN_EXP_F32 -87.0f
#define MAX_EXP_F32 88.0f
#define MIN_EXP_F16 __float2half(11.08f)
#define MAX_EXP_F16 __float2half(-9.70f)

/**
 * @brief Swish kernel
 * 
 * Swish(x) = x * sigmoid(x)
 * 
 * API: 
 *      __hneg
 *      Clamp when using exp (expf, hexp)
 * 
 */

__device__ __forceline__ float sigmoid(float x) {
    float val = fminf(fmaxf(MIN_EXP_F32, x), MAX_EXP_F32);
    return 1.0f / (1 + expf(-val));
}

__device__ __forceinline__ half sigmoid_f16(half x) {
    half val = __hmin(__hmax(MIN_EXP_F16, x), MAX_EXP_F16);
    half one = __float2half(1.0f);
    return __hdiv(one, __hadd(one, hexp(__hneg(val))));
}

__device__ __forceinline__ float swish(float x) {
    return x * sigmoid(x);
}

__global__ void swish_f16(half *a, half *ans, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        ans[idx] = sigmoid_f16(a[idx]);
    }
}

