#include <iostream>

static constexpr int size = 1024 * 35;
static constexpr int thread_per_block = 256;
static constexpr int block_num = std::min(32, (size + thread_per_block - 1) / thread_per_block);

__global__ void dot_product(float *a, float *b, float *partial) {
    // __shared__ is shared within a thread block
    __shared__ float tmp[thread_per_block];     // sum of column of current thread
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    float tmp_sum = 0;
    while (x < size) {
        tmp_sum += a[x] * b[x];
        x += blockDim.x * gridDim.x;
    }
    tmp[threadIdx.x] = tmp_sum;
    __syncthreads();    // wait for all threads to finish
    
    // Reduction starts
    int scale = thread_per_block / 2;
    while (scale > 0) {
        if (threadIdx.x < scale) {
            tmp[threadIdx.x] += tmp[threadIdx.x + scale];
        }
        // ensure that all threads have updated the sum
        // otherwise scale will change, and the index to be updated is wrong
        __syncthreads();
        scale /= 2;
    }

    if (threadIdx.x == 0) {
        partial[blockIdx.x] = tmp[0];
    }
}

int main() {
    float a[size], b[size], res[block_num];
    // dev_partial stores the reduction result of each block
    float *dev_a, *dev_b, *dev_partial;
    float standard_sum = 0;

    for (int i = 0; i < size; ++i) {
        a[i] = i;
        b[i] = i * 2;
        standard_sum += static_cast<float>(a[i]) * b[i];
    }

    cudaMalloc((void **)&dev_a, size * sizeof(float));
    cudaMalloc((void **)&dev_b, size * sizeof(float));
    cudaMalloc((void **)&dev_partial, block_num * sizeof(float));

    cudaMemcpy(dev_a, a, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size * sizeof(float), cudaMemcpyHostToDevice);

    dot_product<<<block_num, thread_per_block>>>(dev_a, dev_b, dev_partial);

    cudaMemcpy(res, dev_partial, block_num * sizeof(float), cudaMemcpyDeviceToHost);

    float sum = 0;
    for (int i = 0; i < block_num; ++i) {
        sum += res[i];
    }

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_partial);

    if (abs(sum - standard_sum) > 1e-6) {
        printf("Error: Sum deviates too much from standard sum\n");
        printf("sum is %f, standard is %f\n", sum, standard_sum);
        exit(1);
    }

    printf("Success: The dot product of vector a and b is %f\n", sum);

    return 0;
}