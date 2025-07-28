#include <iostream>
#include <random>

/** @brief get the histogram of a char array */

static constexpr int SIZE = 1024 * 1024;
static constexpr int RANGE = 256;

__global__ void atomic_add_kernel(int *cnt, unsigned char *array) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    while (x < SIZE) {
        atomicAdd(&cnt[static_cast<int>(array[x])], 1);
        x += stride;
    }
}

__global__ void atomic_add_with_shared_buffer_kernel(int *cnt, unsigned char *array) {
    __shared__ int buf[RANGE];
    buf[threadIdx.x] = 0;
    __syncthreads();
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    while (x < SIZE) {
        atomicAdd(&buf[static_cast<int>(array[x])], 1);
        x += stride;
    }
    __syncthreads();
    atomicAdd(&cnt[threadIdx.x], buf[threadIdx.x]);
}

int main() {
    unsigned char array[SIZE];
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, RANGE - 1);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    
    int answer[RANGE] = {0};

    for (int i = 0; i < SIZE; ++i) {
        array[i] = dis(gen);
        ++answer[static_cast<int>(array[i])];
    }

    int cnt[RANGE];
    unsigned char *dev_array;
    int *dev_cnt;

    cudaMalloc((void **)&dev_array, SIZE * sizeof(unsigned char));
    cudaMalloc((void **)&dev_cnt, RANGE * sizeof(int));
    
    cudaMemcpy(dev_array, array, SIZE * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemset(dev_cnt, 0, RANGE * sizeof(int));

    // atomic_add_kernel<<<16, 256>>>(dev_cnt, dev_array);
    atomic_add_with_shared_buffer_kernel<<<16, 256>>>(dev_cnt, dev_array);

    cudaMemcpy(cnt, dev_cnt, RANGE * sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, end);
    printf("Kernel time: %.3f ms\n", elapsed_time);

    cudaEventDestroy(start);
    cudaEventDestroy(end);
    cudaFree(dev_array);
    cudaFree(dev_cnt);

    for (int i = 0; i < RANGE; ++i) {
        if (cnt[i] != answer[i]) {
            printf("Error: different count on char id %d\n", i);
            printf("Supposed to be %d, but got %d\n", answer[i], cnt[i]);
            exit(1);
        }
    }

    printf("Success: Every char count matches.\n");

    return 0;
}