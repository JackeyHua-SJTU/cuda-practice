#include <iostream>

/** @brief Vector add with PSEUDO multi stream */

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
    int *dev_a1, *dev_b1, *dev_c1;
    int *dev_a2, *dev_b2, *dev_c2;

    cudaHostAlloc((void **)&a, TOTAL * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void **)&b, TOTAL * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void **)&c, TOTAL * sizeof(int), cudaHostAllocDefault);

    for (int i = 0; i < TOTAL; ++i) {
        a[i] = i;
        b[i] = i;
    }

    cudaMalloc((void **)&dev_a1, SIZE * sizeof(int));
    cudaMalloc((void **)&dev_b1, SIZE * sizeof(int));
    cudaMalloc((void **)&dev_c1, SIZE * sizeof(int));
    cudaMalloc((void **)&dev_a2, SIZE * sizeof(int));
    cudaMalloc((void **)&dev_b2, SIZE * sizeof(int));
    cudaMalloc((void **)&dev_c2, SIZE * sizeof(int));

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);

    // What about change the order
    
    for (int i = 0; i < TOTAL; i += 2 * SIZE) {
        cudaMemcpyAsync(dev_a1, a + i, SIZE * sizeof(int), cudaMemcpyHostToDevice, stream1);
        cudaMemcpyAsync(dev_b1, b + i, SIZE * sizeof(int), cudaMemcpyHostToDevice, stream1);
        add<<<32, 256, 0, stream1>>>(dev_a1, dev_b1, dev_c1);
        cudaMemcpyAsync(c + i, dev_c1, SIZE * sizeof(int), cudaMemcpyDeviceToHost, stream1);

        cudaMemcpyAsync(dev_a2, a + i + SIZE, SIZE * sizeof(int), cudaMemcpyHostToDevice, stream2);
        cudaMemcpyAsync(dev_b2, b + i + SIZE, SIZE * sizeof(int), cudaMemcpyHostToDevice, stream2);
        add<<<32, 256, 0, stream2>>>(dev_a2, dev_b2, dev_c2);
        cudaMemcpyAsync(c + i + SIZE, dev_c2, SIZE * sizeof(int), cudaMemcpyDeviceToHost, stream2);
    }

    // an old fashioned way to opimize, deprecated
    // because it can only eliminate the wait bubble, but can not execute in parallel
    // The former method can parallelize D2H and H2D

    // for (int i = 0; i < TOTAL; i += 2 * SIZE) {
    //     cudaMemcpyAsync(dev_a1, a + i, SIZE * sizeof(int), cudaMemcpyHostToDevice, stream1);
    //     cudaMemcpyAsync(dev_a2, a + i + SIZE, SIZE * sizeof(int), cudaMemcpyHostToDevice, stream2);
    //     cudaMemcpyAsync(dev_b1, b + i, SIZE * sizeof(int), cudaMemcpyHostToDevice, stream1);
    //     cudaMemcpyAsync(dev_b2, b + i + SIZE, SIZE * sizeof(int), cudaMemcpyHostToDevice, stream2);
    //     add<<<32, 256, 0, stream1>>>(dev_a1, dev_b1, dev_c1);
    //     add<<<32, 256, 0, stream2>>>(dev_a2, dev_b2, dev_c2);
    //     cudaMemcpyAsync(c + i, dev_c1, SIZE * sizeof(int), cudaMemcpyDeviceToHost, stream1);
    //     cudaMemcpyAsync(c + i + SIZE, dev_c2, SIZE * sizeof(int), cudaMemcpyDeviceToHost, stream2);
    // }
    

    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    cudaEventRecord(end, 0);
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
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaFreeHost(a);
    cudaFreeHost(b);
    cudaFreeHost(b);
    cudaFree(dev_a1);
    cudaFree(dev_b1);
    cudaFree(dev_c1);
    cudaFree(dev_a2);
    cudaFree(dev_b2);
    cudaFree(dev_c2);

    return 0;
}