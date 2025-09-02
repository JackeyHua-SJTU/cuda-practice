#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <half>

// Use index instead of dereference directly
#define FLOAT4(value) (reinterpret_cast<float4 *>(&val)[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&val)[0])

/**
 * @brief element wise add for 1d array
 * 
 * Key points:
 *      float4, half, half2
 *      __hadd, __hadd2 (SIMD)
 *      loop unroll
 *      pack trick and standard border case handling
 * 
 * @note pack is to pack multiple basic type into bigger one
 *       for example, float (32 bits) -> float4
 *                    half (16 bits) -> half2
 *       Fewer instructions on the same memory visit target -> faster.
 *       Also there are SIMD instructions for faster computation.
 */

/**
 *  Grid (ceil(N / 256), 1, 1)
 *  Block(256, 1, 1)
 */
__global__ void elementwise_add_f32(float *a, float *b, float *res, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        res[idx] = a[idx] + b[idx];
    }
}

/**
 *  Float4 version
 * 
 *  Grid (ceil(N / 256), 1, 1)
 *  Block(256 / 4, 1, 1)
 */
__global__ void elementwise_add_f32_4(float *a, float *b, float *res, int N) {
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < N) {
        float4 var_a = FLOAT4(a[idx]);
        float4 var_b = FLOAT4(b[idx]);
        float4 var_res;
        var_res.x = var_a.x + var_b.x;
        var_res.y = var_a.y + var_b.y;
        var_res.z = var_a.z + var_b.z;
        var_res.w = var_a.w + var_b.w;
        FLOAT4(res[idx]) = var_res;
    }
}

/**
 *  Grid (ceil(N / 256), 1, 1)
 *  Block(256, 1, 1)
 *  Half precision
 */
__global__ void elementwise_add_f16(half *a, half *b, half *res, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        res[idx] = __hadd(a[idx], b[idx]);
    }
}

/**
 *  half2 version
 * 
 *  Grid (ceil(N / 256), 1, 1)
 *  Block(256 / 2, 1, 1)
 *  Half precision
 */
__global__ void elementwise_add_f16_2(half *a, half *b, half *res, int N) {
    int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < N) {
        half2 var_a = HALF2(a[idx]);
        half2 var_b = HALF2(b[idx]);
        // half2 var_res = __hadd2(var_a, var_b);
        half2 var_res;
        var_res.x = __hadd(var_a.x, var_b.x);   // better use `__hadd`
        var_res.y = __hadd(var_a.y, var_b.y);
        HALF2(res[idx]) = var_res;
    }
}

/**
 *  Float4 version
 * 
 *  Grid (ceil(N / 256), 1, 1)
 *  Block(256 / 8, 1, 1)
 * 
 *  A wrong way ->
 *      float4 var_a = FLOAT4(a[idx]);
        float4 var_b = FLOAT4(b[idx]);
        
        half2 var_a_x = var_a.x;
        half2 var_b_x = var_b.x;
        half2 var_res_x;
        var_res_x.x = __hadd(var_a_x.x, var_b_x.x);
        var_res_x.y = __hadd(var_a_x.y, var_b_x.y);
 *  
 *  We can not assign a float4 to a half2, either reinterpret_cast, 
 *  or just read a half2 one time.
 */
__global__ void elementwise_add_f16_8(half *a, half *b, half *res, int N) {
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < N) {
        half2 var_a_x = HALF2(a[idx]);
        half2 var_b_x = HALF2(b[idx]);
        half2 var_res_x;
        var_res_x.x = __hadd(var_a_x.x, var_b_x.x);
        var_res_x.y = __hadd(var_a_x.y, var_b_x.y);

        half2 var_a_y = HALF2(a[idx + 2]);
        half2 var_b_y = HALF2(b[idx + 2]);
        half2 var_res_y;
        var_res_y.x = __hadd(var_a_y.x, var_b_y.x);
        var_res_y.y = __hadd(var_a_y.y, var_b_y.y);

        half2 var_a_z = HALF2(a[idx + 4]);
        half2 var_b_z = HALF2(a[idx + 4]);
        half2 var_res_z;
        var_res_z.x = __hadd(var_a_z.x, var_b_z.x);
        var_res_z.y = __hadd(var_a_z.y, var_b_z.y);

        half2 var_a_w = HALF2(a[idx + 6]);
        half2 var_b_w = HALF2(a[idx + 6]);
        half2 var_res_w;
        var_res_w.x = __hadd(var_a_w.x, var_b_w.x);
        var_res_w.y = __hadd(var_a_w.y, var_b_w.y);

        HALF2(res[idx]) = var_res_x;
        HALF2(res[idx + 2]) = var_res_y;
        HALF2(res[idx + 4]) = var_res_w;
        HALF2(res[idx + 6]) = var_res_z;
    }
}

/**
 *  Float4 and pack version, with standard border case handling
 * 
 *  Grid (ceil(N / 256), 1, 1)
 *  Block(256 / 8, 1, 1)
 */
__global__ void elementwise_add_f16_8_pack(half *a, half *b, half *res, int N) {
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < N) {
        half reg_a[8], reg_b[8], reg_res[8];    // IMPORTANT, easy for loop unrolling
        FLOAT4(reg_a[0]) = FLOAT4(a[idx]);
        FLOAT4(reg_b[0]) = FLOAT4(b[idx]);

        #pragma unroll
        for (int i = 0; i < 8; i += 2) {
            HALF2(reg_res[i]) = __hadd2(HALF2(reg_a[i]), HALF2(reg_b[i]));
        }
        if (idx + 7 < N) {
            FLOAT4(res[idx]) = FLOAT4(reg_res[0]);
        } else {
            for (int i = idx; i < N; ++i) {
                res[i] = reg_res[i - idx];
            }
        }
    }
}

