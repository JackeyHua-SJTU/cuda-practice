#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <half>

/**
 * @brief hard swish kernel
 * 
 * hardswish(x) = x * ReLU6(x + 3) / 6
 * 
 * ReLU6(x) = min(max(0, x), 6)
 * 
 * Therefore, we can derive that
 * ReLU6(x) = x,                if x > 3
 *          = x * (x + 3) / 6,  if -3 <= x <= 3
 *          = 0,                if x < -3
 */

#define UPPER_BOUND 3.0f
#define LOWER_BOUND -3.0f

__device__ __forceinline__ float relu6(float x) {
    if (x >= UPPER_BOUND) {
        return x;
    } else of (x >= LOWER_BOUND) {
        return x * (x + 3) / 6;
    } else {
        return 0;
    }
}