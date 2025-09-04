#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <half>

#define ALPHA 1.0f

#define FLOAT4(value) (reinterpret_cast<float4 *>(&value)[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&value)[0])

#define GELU_CONST 0.044715f
#define HALF_HALF       __float2half(0.5f)
#define HALF_1          __float2half(1.0f)
#define HALF_2          __float2half(2.0f)
#define HALF_PI         __float2half(CUDART_PI_F)
#define HALF_CONST_GELU __float2half(GELU_CONST)

#define MIN_EXP_F32 -87.0f
#define MAX_EXP_F32 88.0f

/**
 * @brief GeLU
 * 
 * Accurate: GeLU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
 * 
 * Inaccurate: GeLu(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 x^3)))
 * 
 * Tanh(x) = (e^2x - 1) / (e^2x + 1)
 * 
 * API:
 *      erff - error function (https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv43erfd)
 *      rsqrtf - reciprocal of square root of float type
 *      tanhf
 *      hsqrt, htanh
 */

__device__ __forceinline__ float gelu_accurate(float x) {
    return 0.5f * x * (1 + erff(x * rsqrtf(2.0f)));
}

__device__ __forceinline__ float gelu_approximate(float x) {
    return 0.5f * x * (1 + 
        tanhf(sqrtf(2.0f / CUDART_PI_F) * (x + GELU_CONST * x * x * x)));
}

__device__ __forceinline__ half gelu_approximate_f16(half x) {
    half cube = x * x * x;
    return HALF_HALF * x *
            (HALF_1  + 
                htanh(hsqrt(HALF_2 / HALF_PI) * 
                    (x + HALF_CONST_GELU * cube)));
}

__global__ void gelu_f32(float *a, float *ans, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = a[idx];
        val = fminf(fmaxf(MIN_EXP_F32, val), MAX_EXP_F32);
        ans[idx] = gelu_accurate(val);
    }
}