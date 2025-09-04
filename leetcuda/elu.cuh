#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <half>

#define ALPHA 1.0f

#define FLOAT4(value) (reinterpret_cast<float4 *>(&value)[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&value)[0])

/**
 * @brief ELU kernel
 * 
 * Elu(x) = x if x >= 0
 *          a * (exp(x) - 1) if x < 0
 * 
 * API:
 *      __forceinline__
 *      __hge
 *      __hsub
 *      __hmul
 */

__device__ __forceinline__ float elu(float x) {
    if (x >= 0.0f) {
        return x;
    }
    return ALPHA * (expf(x) - 1.0f);
}

__device__ __forceinline__ half elu_f16(half x) {
    half zero = __float2half(0.0f);
    return __hge(x, zero)
                ? half 
                : __hmul(__float2half(ALPHA), __hsub(hexp(x), __float2half(1.0f)));
}

__global__ void elu_f32(float *a, float *ans, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        ans[idx] = elu(a[idx]);
    }
}

__global__ void elu_f32_4(float *a, float *ans, int N) {
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < N) {
        float4 val = FLOAT4(a[idx]);
        float4 result;
        result.x = elu(val.x);
        result.y = elu(val.y);
        result.w = elu(val.w);
        result.z = elu(val.z);
        FLOAT4(ans[idx]) = result;
    }
}

__global__ void elu_f16(half *a, half *ans, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        ans[idx] = elu_f16(a[idx]);
    }
}

__global__ void elu_f16_2(half *a, half *ans, int N) {
    int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < N) {
        half2 val = HALF2(a[idx]);
        half2 res;
        res.x = elu_f16(val.x);
        res.y = elu_f16(val.y);
        HALF2(ans) = res;
    }
}

__global__ void elu_f16_8_pack(half *a, half *ans, int N) {
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    half reg_a[8], reg_ans[8];
    FLOAT4(reg_a[0]) = FLOAT4(a[idx]);
    
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        reg_ans[i] = elu_f16(reg_a[i]);
    }
    
    if (idx + 7 < N) {
        FLOAT4(ans[idx]) = FLOAT4[reg_ans[0]];
    } else {
        for (int i = idx; i < N; ++i) {
            ans[i] = reg_ans[i - idx];
        }
    }
}
