#include <iostream>
#include <thread>
#include <vector>


static constexpr int SIZE = 30 * 1024 * 1024;
static constexpr int GPU_COUNT = 4;

struct per_gpu_info {
    int device_id;
    int size;
    float *a;
    float *b;
    float *ret_val;
};

__global__ void add_kernel(float *a, float *b, float *ans, int size) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    float tmp = 0;
    while (x < size) {
        tmp += a[x] + b[x];
        x += stride;
    }
    atomicAdd(ans, tmp);
}

void launch(per_gpu_info *info) {
    cudaSetDevice(info->device_id);
    float *dev_a, *dev_b, *dev_ans;
    
    cudaMalloc((void **)&dev_a, info->size * sizeof(float));
    cudaMalloc((void **)&dev_b, info->size * sizeof(float));
    cudaMalloc((void **)&dev_ans, sizeof(float));

    cudaMemcpy(dev_a, info->a, info->size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, info->b, info->size * sizeof(float), cudaMemcpyHostToDevice);
    
    add_kernel<<<16, 256>>>(dev_a, dev_b, dev_ans, info->size);

    cudaMemcpy(info->ret_val, dev_ans, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_ans);
}

int main() {
    float *a, *b, *ans;
    float correct_result;
    a = (float *)malloc(SIZE * sizeof(float));
    b = (float *)malloc(SIZE * sizeof(float));
    ans = (float *)malloc(GPU_COUNT * sizeof(float));

    for (int i = 0; i < SIZE; ++i) {
        a[i] = i;
        b[i] = i;
        correct_result += 2 * i;
    }

    per_gpu_info handle[GPU_COUNT];
    for (int i = 0; i < GPU_COUNT; ++i) {
        handle[i].device_id = i;
        handle[i].size = SIZE / GPU_COUNT;
        handle[i].ret_val = ans + i;
        handle[i].a = a + i * handle[i].size;
        handle[i].b = b + i * handle[i].size;
    }

    std::vector<std::thread> jobs;

    for (int i = 0; i < GPU_COUNT; ++i) {
        jobs.emplace_back(launch, &handle[i]);
    }

    for (auto &t : jobs) {
        t.join();
    }
    
    float result = 0;
    for (int i = 0; i < GPU_COUNT; ++i) {
        result += ans[i];
    }
    printf("answer is %.3f\n", result);
    printf("correct answer is %.3f\n", correct_result);

    free(a);
    free(b);
    free(ans);
    return 0;
}