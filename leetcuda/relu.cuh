#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <half>

#define FLOAT4(value) (reinterpret_cast<float4 *>(&value)[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&value)[0])

/**
 * @brief ReLU kernel
 * 
 * Relu(x) = max(0, x)
 * 
 * API:
 *  __half2
 *  Initialization of half2 type
 */

__global__ void relu_f16_8_pack(half *a, half *ans, int N) {
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    const half zero = __float2half(0.0f);

    half reg_a[8], reg_ans[8];
    FLOAT4(reg_a[0]) = FLOAT4(a[idx]);

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        reg_ans[i] = __hmax(zero, reg_a[i]);
    }

    // alternative __half2 methods
    #pragma unroll
    for (int i = 0; i < 8; i += 2) {
        HALF2(reg_ans[i]) = __hmax2({zero, zero},
                                {reg_a[i], reg_a[i + 1]});
    }

    if (idx + 7 < N) {
        FLOAT4(ans[idx]) = FLOAT4(reg_ans[0]);
    } else {
        for (int i = idx; i < N; ++i) {
            ans[i] = reg_ans[i - idx];
        }
    }
}