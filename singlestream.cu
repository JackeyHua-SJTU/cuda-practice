#include <iostream>

/** @brief Vector add with single stream 
 * 
 *  @note
 *  1. In stream and async data moving, we should use page-locked memory
 *  2. Use `cudaStreamSynchronize` to wait for certain stream to finish
 *  3. Blocking the computation. Deal with just a block in an iteration
*/

static constexpr int SIZE = 1024 * 1024;
static constexpr int TOTAL = 20 * SIZE;

__global__ void add(int *a, int *b, int *c) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    while (x < SIZE) {
        c[x] = a[x] + b[x];
        x += stride;
    }
}

int main() {
    int *a, *b, *c;
    int *dev_a, *dev_b, *dev_c;

    cudaHostAlloc((void **)&a, TOTAL * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void **)&b, TOTAL * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void **)&c, TOTAL * sizeof(int), cudaHostAllocDefault);

    for (int i = 0; i < TOTAL; ++i) {
        a[i] = i;
        b[i] = i;
    }

    cudaMalloc((void **)&dev_a, SIZE * sizeof(int));
    cudaMalloc((void **)&dev_b, SIZE * sizeof(int));
    cudaMalloc((void **)&dev_c, SIZE * sizeof(int));

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, stream);
    
    for (int i = 0; i < TOTAL; i += SIZE) {
        cudaMemcpyAsync(dev_a, a + i, SIZE * sizeof(int), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(dev_b, b + i, SIZE * sizeof(int), cudaMemcpyHostToDevice, stream);
        add<<<32, 256, 0, stream>>>(dev_a, dev_b, dev_c);
        cudaMemcpyAsync(c + i, dev_c, SIZE * sizeof(int), cudaMemcpyDeviceToHost, stream);
    }

    cudaStreamSynchronize(stream);

    cudaEventRecord(end, stream);
    cudaEventSynchronize(end);

    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, end);
    
    printf("Total time: %.3f ms\n", elapsed_time);
    
    for (int i = 0; i < SIZE; ++i) {
        if (c[i] != i * 2) {
            printf("Error value of c array on index %d\n", i);
            exit(1);
        }
    }
    printf("Success: Every index has correct answer\n");

    cudaEventDestroy(start);
    cudaEventDestroy(end);
    cudaStreamDestroy(stream);
    cudaFreeHost(a);
    cudaFreeHost(b);
    cudaFreeHost(b);
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    return 0;
}