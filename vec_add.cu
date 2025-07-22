#include <iostream>

static constexpr int SIZE = 300 * 1024;

__global__ void simple_add(int *a, int *b, int *c) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    while (index < SIZE) {
        c[index] = a[index] + b[index];
        index += blockDim.x * gridDim.x;
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
    // dim3 grid((SIZE + 15) / 16, 16);  // 多维度必须这么写
    simple_add<<<128, 128>>>(var_a, var_b, dev_p);
    cudaMemcpy(result, dev_p, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dev_p);
    cudaFree(var_a);
    cudaFree(var_b);
    for (int i = 0; i < SIZE; ++i) {
        if (a[i] + b[i] != result[i]) {
            printf("Error Calc result\n");
            exit(1);
        }
    }
    printf("Every calc result is correct!\n");
    return 0;
}