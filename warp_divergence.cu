#include <iostream>

/** @brief Benchmark the performance of warp divergence */

static constexpr int SIZE = 30 * 1024 * 1024;
static constexpr int WARP_SIZE = 32;

__global__ void kernel_slow(int *a, int *b, int *result) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    while (x < SIZE) {
        bool flag = x % 2;
        if (flag) {
            result[x] = a[x] + b[x];
        } else {
            result[x] = a[x] - b[x];
        }
        x += stride;
    }
}

__global__ void kernel_fast(int *a, int *b, int *result) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    while (x < SIZE) {
        // So that all threads in the warp will choose the same clause
        // No divergence
        if ((x / WARP_SIZE) % 2) {
            result[x] = a[x] + b[x];
        } else {
            result[x] = a[x] - b[x];
        }
        x += stride;
    }
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    // usually warp size is 32 threads/warp
    printf("Threads count inside a warp is %d\n", prop.warpSize);
    
    int *a, *b, *result;
    int *dev_a, *dev_b, *dev_result;
    cudaEvent_t start, end;
    float elapsed_time;

    cudaEventCreate(&start);
    cudaEventCreate(&end);

    a = (int *)malloc(SIZE * sizeof(int));
    b = (int *)malloc(SIZE * sizeof(int));
    result = (int *)malloc(SIZE * sizeof(int));

    for (int i = 0; i < SIZE; ++i) {
        a[i] = i;
        b[i] = i;
    }

    cudaMalloc((void **)&dev_a, SIZE * sizeof(int));
    cudaMalloc((void **)&dev_b, SIZE * sizeof(int));
    cudaMalloc((void **)&dev_result, SIZE * sizeof(int));

    cudaMemcpy(dev_a, a, SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, SIZE * sizeof(int), cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    // kernel launch
    kernel_slow<<<16, 128>>>(dev_a, dev_b, dev_result);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, start, end);

    printf("Slow version takes %.3f ms \n", elapsed_time);  // takes about 7.07 ms

    cudaMemset(dev_result, 0, SIZE * sizeof(int));

    cudaEventRecord(start);
    // kernel launch
    kernel_fast<<<16, 128>>>(dev_a, dev_b, dev_result);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, start, end);

    printf("Fast version takes %.3f ms \n", elapsed_time);  // takes about 6.73 ms

    free(a);
    free(b);
    free(result);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_result);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    return 0;
}