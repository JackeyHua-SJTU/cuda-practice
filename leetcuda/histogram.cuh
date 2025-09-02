#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <half>

#define INT4(value) (reinterpret_cast<int4 *>(&value)[0])

/**
 *  @brief Histogram kernel
 * 
 *  Not quite suitable for using shared memory
 *  because the range of numbers is too large.
 */

/**
 *  Grid (ceil(N / 256), 1, 1)
 *  Block(256, 1, 1)
 */
__global__ void histogram_i32(int *a, int *res, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        atomicAdd(&res[a[idx]], 1);
    }
}

/**
 *  Grid (ceil(N / 256), 1, 1)
 *  Block(256 / 4, 1, 1)
 */
__global__ void histogram_i32_4(int *a, int *res, int N) {
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < N) {
        int4 var_a = INT4(a[idx]);
        
        atomicAdd(&res[var_a.x], 1);
        
        if (idx + 1 < N) {
            atomicAdd(&res[var_a.y], 1);
        }

        if (idx + 2 < N) {
            atomicAdd(&res[var_a.z], 1);
        }

        if (idx + 3 < N) {
            atomicAdd(&res[var_a.w], 1);
        }
    }
}
