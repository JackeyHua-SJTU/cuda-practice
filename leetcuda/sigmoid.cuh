#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <half>

// in case of overflow
#define MIN_EXP_F32 -87.0f
#define MAX_EXP_F32 88.0f
#define MIN_EXP_F16 __float2half(11.08f)
#define MAX_EXP_F16 __float2half(-9.70f)

#define FLOAT4(value) (reinterpret_cast<float4 *>(&value)[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&value)[0])

/**
 * @brief Sigmoid kernel
 * 
 * Sigmoid(x) = 1 / (1 + exp(-x))
 * Clamp to [MIN_EXP_F32, MAX_EXP_F32] or [MIN_EXP_F16, MAX_EXP_F16]
 *  in case of float/half type overflow.
 * 
 * API:
 *  __hadd, __hdiv, __hmax, __hmin
 *  __float2half (for data type conversion)
 *  hexp
 *  fmaxf, fminf
 *  expf
 * 
 */

__global__ void sigmoid_f32(float *a, float *res, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float v = a[idx];
        v = fminf(fmaxf(v, MIN_EXP_F32), MAX_EXP_F32);
        res[idx] = 1.0f / (1.0f + expf(-v));
    }
}

__global__ void sigmoid_f32_4(float *a, float *res, int N) {
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < N) {
        float4 data = FLOAT4(a[idx]);
        float4 res;

        float var_x = data.x;
        var_x = fminf(fmaxf(var_x, MIN_EXP_F32), MAX_EXP_F32);
        res.x = 1.0f / (1.0f + expf(-var_x));

        float var_y = data.y;
        var_y = fminf(fmaxf(var_y, MIN_EXP_F32), MAX_EXP_F32);
        res.y = 1.0f / (1.0f + expf(-var_y));

        float var_z = data.z;
        var_z = fminf(fmaxf(var_z, MIN_EXP_F32), MAX_EXP_F32);
        res.z = 1.0f / (1.0f + expf(-var_z));

        float var_w = data.w;
        var_w = fminf(fmaxf(var_w, MIN_EXP_F32), MAX_EXP_F32);
        res.w = 1.0f / (1.0f + expf(-var_w));

        FLOAT4(res[idx]) = res;
    }
}

__global__ void sigmoid_f16(half *a, half *res, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const half val = __float2half(1.0f);
    if (idx < N) {
        half v = a[idx];
        v = __hmin(__hmax(v, MIN_EXP_F16), MAX_EXP_F16);
        res[idx] = __hdiv(val, val + hexp(-v));
    }
}

__global__ void sigmoid_f16_2(half *a, half *res, int N) {
    int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    const half val = __float2half(1.0f);
    if (idx < N) {
        half2 v = HALF2(a);
        half2 result;
        v.x = __hmin(__hmax(v.x, MIN_EXP_F16), MAX_EXP_F16);
        v.y = __hmin(__hmax(v.y, MIN_EXP_F16), MAX_EXP_F16);
        result.x = __hdiv(val, val + hexp(-v.x));
        result.y = __hdiv(val, val + hexp(-v.y));
        HALF2(res[idx]) = result;
    }
}

__global__ void sigmoid_f16_8_pack(half *a, half *res, int N) {
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    const half val = __float2half(1.0f);
    half reg_a[8], reg_res[8];
    FLOAT4(reg_a[0]) = FLOAT4(a[idx]);

    #pragma unroll
    for (int i = 0; i < 8; i += 1) {
        reg_a[i] = __hmin(__hmax(reg_a[i], MIN_EXP_F16), MAX_EXP_F16);
        res[i] = __hdiv(val, val + hexp(-reg_a[i]));
    }

    if (idx + 7 < N) {
        FLOAT4(res[idx]) = FLOAT4(reg_res[0]);
    } else {
        for (int i = idx; i < N; ++i) {
            res[i] = reg_res[i - idx];
        }
    }
}