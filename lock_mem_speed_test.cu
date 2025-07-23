#include <iostream>

/** @brief test the performance of copying malloc and cudaHostAlloc host memory to GPU 
 * 
 * Page-locked memory is much faster than noraml one because it is fixed in memory, and 
 * can not be swapped out to disk.
*/

static constexpr int SIZE = 30 * 1024 * 1024;

int main() {
    float elapsed_time;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    int *host_mem = (int *)malloc(SIZE * sizeof(int));
    int *locked_host_mem;   // can not be paged and swapped into disk, always in memory
    cudaHostAlloc((void **)&locked_host_mem, SIZE * sizeof(int), cudaHostAllocDefault);

    int *dev_host_mem, *dev_locked_host_mem;
    cudaMalloc((void **)&dev_host_mem, SIZE * sizeof(int));
    cudaMalloc((void **)&dev_locked_host_mem, SIZE * sizeof(int));

    cudaEventRecord(start);
    cudaMemcpy(dev_host_mem, host_mem, SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(host_mem, dev_host_mem, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    cudaEventElapsedTime(&elapsed_time, start, end);
    printf("Normal malloc takes %.3f ms\n", elapsed_time);

    cudaEventRecord(start);
    cudaMemcpy(dev_locked_host_mem, locked_host_mem, SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(locked_host_mem, dev_locked_host_mem, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    cudaEventElapsedTime(&elapsed_time, start, end);
    printf("Locked malloc takes %.3f ms\n", elapsed_time);

    cudaEventDestroy(start);
    cudaEventDestroy(end);
    cudaFree(dev_host_mem);
    cudaFree(dev_locked_host_mem);
    cudaFreeHost(locked_host_mem);  // special free function
    free(host_mem);

    return 0;
}