#include <iostream>

/** @brief compare the performance of in-place vec reduce(Add op) 
 *      and optimize it by reducing warp divergence
 */

// TODO: Why slow kernel has better performance when SIZE reaches 50 * 1024 * 1024

static constexpr int SIZE = 10 * 1024 * 1024;
static constexpr int blocknum = 32;   // divide into `blocknum` blocks
static constexpr int thread_per_block = 512;

/**
 *  @brief In-place add
 *      Always reduce the consecutive two valid entry, until there is only one
 *  @note let per_block_size be in the form of 2 ** k
 */

__global__ void slow_kernel(int *arr, int *ans) {
    int base_index = threadIdx.x + blockIdx.x * blockDim.x;
    int x = base_index;
    int stride = blockDim.x * gridDim.x;
    int temp = 0;
    // first accumulate different grid group
    // This is to handle arbitrary long array
    while (x < SIZE) {
        temp += arr[x];
        x += stride; 
    }
    arr[base_index] = temp;
    __syncthreads();
    for (int gap = 1; gap <= thread_per_block / 2; gap *= 2) {
        // The target of every thread is base_index
        // Some will pass and some will pause
        if (base_index % (2 * gap) == 0) {
            arr[base_index] += arr[base_index + gap];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        ans[blockIdx.x] = arr[base_index];
    }
}

__global__ void fast_kernel(int *arr, int *ans) {
    int base = blockIdx.x * blockDim.x;
    int base_index = threadIdx.x + base;
    int x = base_index;
    int stride = blockDim.x * gridDim.x;
    int temp = 0;
    // first accumulate different grid group
    while (x < SIZE) {
        temp += arr[x];
        x += stride;
    }
    arr[base_index] = temp;
    __syncthreads();
    for (int gap = 1; gap <= thread_per_block / 2; gap *= 2) {
        int index = 2 * gap * threadIdx.x;
        // The target of every thread is index
        // Unless it exceeds the upper bound, we can assure that 
        //  they are all valid
        if (index < thread_per_block) {
            arr[base + index] += arr[base + index + gap];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        ans[blockIdx.x] = arr[base_index];
    }
}

__global__ void slow_kernel_shared(int *arr, int *ans) {
    __shared__ int temp_arr[thread_per_block];
    int base_index = threadIdx.x + blockIdx.x * blockDim.x;
    int x = base_index;
    int stride = blockDim.x * gridDim.x;
    int temp = 0;
    // first accumulate different grid group
    // This is to handle arbitrary long array
    while (x < SIZE) {
        temp += arr[x];
        x += stride; 
    }
    temp_arr[threadIdx.x] = temp;
    __syncthreads();
    for (int gap = 1; gap <= thread_per_block / 2; gap *= 2) {
        // The target of every thread is base_index
        // Some will pass and some will pause
        if (base_index % (2 * gap) == 0) {
            temp_arr[threadIdx.x] += temp_arr[threadIdx.x + gap];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        ans[blockIdx.x] = temp_arr[0];
    }
}

__global__ void fast_kernel_shared(int *arr, int *ans) {
    __shared__ int temp_arr[thread_per_block];
    int base_index = threadIdx.x + blockIdx.x * blockDim.x;
    int x = base_index;
    int stride = blockDim.x * gridDim.x;
    int temp = 0;
    // first accumulate different grid group
    while (x < SIZE) {
        temp += arr[x];
        x += stride;
    }
    temp_arr[threadIdx.x] = temp;
    __syncthreads();
    for (int gap = 1; gap <= thread_per_block / 2; gap *= 2) {
        int index = 2 * gap * threadIdx.x;
        // The target of every thread is index
        // Unless it exceeds the upper bound, we can assure that 
        //  they are all valid
        if (index < thread_per_block) {
            temp_arr[index] += temp_arr[index + gap];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        ans[blockIdx.x] = temp_arr[0];
    }
}

int main() {
    int *a, *per_block;
    int *dev_a, *dev_per_block;
    cudaEvent_t start, end;
    float elapsed_time;

    cudaEventCreate(&start);
    cudaEventCreate(&end);

    a = (int *)malloc(SIZE * sizeof(int));
    per_block = (int *)calloc(blocknum, sizeof(int));

    cudaMalloc((void **)&dev_a, SIZE * sizeof(int));
    cudaMalloc((void **)&dev_per_block, blocknum * sizeof(int));

    for (int i = 0; i < SIZE; ++i) {
        a[i] = i;
    }

    cudaMemcpy(dev_a, a, SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(dev_per_block, 0, blocknum * sizeof(int));

    cudaEventRecord(start);
    slow_kernel<<<blocknum, thread_per_block>>>(dev_a, dev_per_block);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, start, end);
    printf("Slower kernel takes %.3f ms\n", elapsed_time);


    cudaMemset(dev_per_block, 0, blocknum * sizeof(int));
    cudaEventRecord(start);
    fast_kernel<<<blocknum, 128>>>(dev_a, dev_per_block);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, start, end);
    printf("Faster kernel takes %.3f ms\n", elapsed_time);

    cudaMemset(dev_per_block, 0, blocknum * sizeof(int));
    cudaEventRecord(start);
    slow_kernel_shared<<<blocknum, thread_per_block>>>(dev_a, dev_per_block);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, start, end);
    printf("Slower kernel with shared memory takes %.3f ms\n", elapsed_time);


    cudaMemset(dev_per_block, 0, blocknum * sizeof(int));
    cudaEventRecord(start);
    fast_kernel_shared<<<blocknum, 128>>>(dev_a, dev_per_block);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, start, end);
    printf("Faster kernel with shared memory takes %.3f ms\n", elapsed_time);

    free(a);
    free(per_block);
    cudaFree(dev_a);
    cudaFree(dev_per_block);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    return 0;
}