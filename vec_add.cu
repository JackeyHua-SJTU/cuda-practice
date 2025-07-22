#include <iostream>

static constexpr int SIZE = 256;

__global__ void simple_add(int *a, int *b, int *c) {
    int x = blockIdx.x;
    int y = blockIdx.y;
    int index = y * gridDim.x + x;
    if (index < SIZE) {
        c[index] = a[index] + b[index];
    }
}

int main() {
    int result[SIZE];
    int *dev_p;
    int count;
    
    cudaGetDeviceCount(&count);
    printf("Total GPU count: %d\n", count);

    cudaDeviceProp prop;
    // get the property of device 0
    cudaGetDeviceProperties(&prop, 0);
    printf("Device 0 (%s) has warp size %d\n", prop.name, prop.warpSize);
    printf("    Major: %d, Minor: %d\n", prop.major, prop.minor);
    printf("    max thread per block: %d\n", prop.maxThreadsPerBlock);

    int a[SIZE], b[SIZE];
    for (int i = 0; i < SIZE; ++i) {
        a[i] = i + 1;
        b[i] = 2 * (i + 1);
    }
    int *var_a, *var_b;
    cudaMalloc((void **)&var_a, SIZE * sizeof(int));
    cudaMalloc((void **)&var_b, SIZE * sizeof(int));
    cudaMemcpy(var_a, a, SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(var_b, b, SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&dev_p, SIZE * sizeof(int));
    dim3 grid(16, 16);  // 多维度必须这么写
    simple_add<<<grid, 1>>>(var_a, var_b, dev_p);
    cudaMemcpy(result, dev_p, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dev_p);
    cudaFree(var_a);
    cudaFree(var_b);
    for (int i = 0; i < SIZE; ++i) {
        printf("answer[%d] is %d\n", i, result[i]);
    }

    return 0;
}