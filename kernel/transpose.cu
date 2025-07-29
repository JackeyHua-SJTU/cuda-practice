#include <iostream>

static constexpr int ROW = 256;
static constexpr int COL = 256;
static constexpr int WARP_SIZE = 16;
static constexpr int SIZE = ROW * COL;
static constexpr int GRIDNUM = 64;
static constexpr int BLOCKNUM = (SIZE + GRIDNUM - 1) / GRIDNUM;

/**
 *  @brief Matrix Transpose Kernel
 *  
 *  1. Write is more expensive than read. Optimize write first (by row write).
 *  2. globalX and globalY are always from the perspective of source matrix
 * 
 *  @todo support arbitrary size matrix
 */ 

// read by row, write by column
__global__ void transpose_rr_wc(int *src, int *ans, const int r, const int c) {
    int globalIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int row = globalIdx / c;
    int col = globalIdx % c;

    if (globalIdx < r * c) {
        ans[col * r + row] = src[globalIdx];
    }
}

// read by column, write by row
__global__ void transpose_rc_wr(int *src, int *ans, const int r, const int c) {
    int globalIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int row = globalIdx / r;
    int col = globalIdx % r;

    // write coalesced, much faster
    if (globalIdx < r * c) {
        ans[globalIdx] = src[col * c + row];
    }
}

// read by column, write by row, with int4
__global__ void transpose_rc_wr_f4(int *src, int *ans, const int r, const int c) {

    int globalIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int row = (globalIdx * 4) / r;
    int col = (globalIdx * 4) % r;

    // write coalesced, much faster
    if (row < c && col + 3 < r) {
        int4 val;
        val.x = src[col * c + row];
        val.y = src[(col + 1) * c + row];
        val.z = src[(col + 2) * c + row];
        val.w = src[(col + 3) * c + row];
        reinterpret_cast<int4 *>(ans)[globalIdx] = val;
    }
}

// read by column, write by row, with 1d shared buffer
__global__ void transpose_rc_wr_1d_shared_buf(int *src, int *ans, const int r, const int c) {
    __shared__ int buf[BLOCKNUM];
    int globalIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int row = globalIdx / r;
    int col = globalIdx % r;

    // write coalesced, much faster
    if (globalIdx < r * c) {
        buf[threadIdx.x] = src[col * c + row];
    }
    __syncthreads();
    ans[globalIdx] = buf[threadIdx.x];
}

// read by row, write by column, 2d grid/block
__global__ void transpose_rr_wc_2d(int *src, int *ans, const int r, const int c) {
    int globalX = threadIdx.x + blockIdx.x * blockDim.x;
    int globalY = threadIdx.y + blockIdx.y * blockDim.y;
    if (globalX < c && globalY < r) {
        ans[globalX * r + globalY] = src[globalY * c + globalX];
    }
}

// read by column, write by row, 2d grid/block
__global__ void transpose_rc_wr_2d(int *src, int *ans, const int r, const int c) {
    int globalX = threadIdx.x + blockIdx.x * blockDim.x;
    int globalY = threadIdx.y + blockIdx.y * blockDim.y;

    if (globalX < r && globalY < c) {
        ans[globalY * r + globalX] = src[globalX * c + globalY];
    }
}

// read by row, write by column, 2d grid/block, int4
// compress the src by row, ans by col
__global__ void transpose_rr_wc_2d_int4(int *src, int *ans, const int r, const int c) {
    int globalX = threadIdx.x + blockIdx.x * blockDim.x;
    int globalY = threadIdx.y + blockIdx.y * blockDim.y;
    if (globalX * 4 + 3 < c && globalY < r) {
        int4 val = reinterpret_cast<int4 *>(src)[globalY * c / 4 + globalX];
        ans[globalX * 4 * r + globalY] = val.x;
        ans[(globalX * 4 + 1) * r + globalY] = val.y;
        ans[(globalX * 4 + 2) * r + globalY] = val.z;
        ans[(globalX * 4 + 3) * r + globalY] = val.w;
    }
}

// read by column, write by row, 2d grid/block, int4
// compress the ans by row, src by col
__global__ void transpose_rc_wr_2d_int4(int *src, int *ans, const int r, const int c) {
    int globalX = threadIdx.x + blockIdx.x * blockDim.x;
    int globalY = threadIdx.y + blockIdx.y * blockDim.y;

    if (globalX < c && globalY * 4 + 3 < r) {
        int4 val;
        val.x = src[globalX + (globalY * 4) * c];
        val.y = src[globalX + (globalY * 4 + 1) * c];
        val.z = src[globalX + (globalY * 4 + 2) * c];
        val.w = src[globalX + (globalY * 4 + 3) * c];
        reinterpret_cast<int4 *>(ans)[globalX * r / 4 + globalY] = val;
    }
}

// read by column, write by row, 2d grid/block, int4
// compress the ans by row, src by col
// use shared buffer to speed up
__global__ void transpose_rc_wr_2d_int4_shared_buf(int *src, int *ans, const int r, const int c) {
    int globalX = threadIdx.x + blockIdx.x * blockDim.x;
    int globalY = threadIdx.y + blockIdx.y * blockDim.y;
    int localX = threadIdx.x;
    int localY = threadIdx.y;

    __shared__ int buf[WARP_SIZE * 4][WARP_SIZE];
    if (globalX < c && globalY * 4 + 3 < r) {
        int4 val;
        val.x = src[globalX + (globalY * 4) * c];
        val.y = src[globalX + (globalY * 4 + 1) * c];
        val.z = src[globalX + (globalY * 4 + 2) * c];
        val.w = src[globalX + (globalY * 4 + 3) * c];
        buf[localX * 4][localY] = val.x;
        buf[localX * 4 + 1][localY] = val.y;
        buf[localX * 4 + 2][localY] = val.z;
        buf[localX * 4 + 3][localY] = val.w;
    }
    __syncthreads();
    if (globalX < c && globalY * 4 + 3 < r) {
        ans[globalX * r + globalY * 4] = buf[localX * 4][localY];
        ans[globalX * r + globalY * 4 + 1] = buf[localX * 4 + 1][localY];
        ans[globalX * r + globalY * 4 + 2] = buf[localX * 4 + 2][localY];
        ans[globalX * r + globalY * 4 + 3] = buf[localX * 4 + 3][localY];
    }

}


void print(int *src, int *cur, int row, int col) {
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            printf("%d ", src[i * col + j]);
        }
        printf("\n");
    }
    
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            printf("%d ", cur[i * col + j]);
        }
        printf("\n");
    }
}

bool verify(int *src, int *cur, int row, int col) {
    #ifdef DEBUG
    print(src, cur, row, col);
    #endif
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            if (src[i * col + j] != cur[j * row + i]) {
                printf("Diff on src<%d, %d>, should be %d, but is %d\n", i, j, src[i * col + j], cur[j * row + i]);
                return false;
            }
        }
    }
    return true;
}

int main() {
    int *a, *res;
    int *dev_a, *dev_res;
    a = (int *)malloc(SIZE * sizeof(int));
    res = (int *)malloc(SIZE * sizeof(int));

    memset(res, 0, SIZE * sizeof(int));
    for (int i = 0; i < SIZE; ++i) {
        a[i] = i;
    }

    cudaEvent_t start, end;
    float elapsed_time;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaMalloc((void **)&dev_a, SIZE * sizeof(int));
    cudaMalloc((void **)&dev_res, SIZE * sizeof(int));
    
    cudaMemcpy(dev_a, a, SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaEventRecord(start);
    transpose_rr_wc<<<64, (SIZE + 63) / 64>>>(dev_a, dev_res, ROW, COL);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, start, end);

    cudaMemcpy(res, dev_res, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    if (verify(a, res, ROW, COL) == false) {
        printf("Wrong answer: kernel transpose_rr_wc\n");
        exit(1);
    }

    printf("Success: kernel transpose_rr_wc takes %.3f ms\n", elapsed_time);

    cudaMemset(dev_res, 0, SIZE * sizeof(int));
    cudaEventRecord(start);
    transpose_rc_wr<<<64, (SIZE + 63) / 64>>>(dev_a, dev_res, ROW, COL);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, start, end);

    cudaMemcpy(res, dev_res, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    if (verify(a, res, ROW, COL) == false) {
        printf("Wrong answer: kernel transpose_rc_wr\n");
        exit(1);
    }

    printf("Success: kernel transpose_rc_wr takes %.3f ms\n", elapsed_time);


    cudaMemset(dev_res, 0, SIZE * sizeof(int));
    cudaEventRecord(start);
    transpose_rc_wr_f4<<<64, (SIZE + 255) / 64 / 4>>>(dev_a, dev_res, ROW, COL);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, start, end);

    cudaMemcpy(res, dev_res, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    if (verify(a, res, ROW, COL) == false) {
        printf("Wrong answer: kernel transpose_rc_wr_f4\n");
        exit(1);
    }

    printf("Success: kernel transpose_rc_wr_f4 takes %.3f ms\n", elapsed_time);

    cudaMemset(dev_res, 0, SIZE * sizeof(int));
    cudaEventRecord(start);
    transpose_rc_wr_1d_shared_buf<<<64, (SIZE + 63) / 64>>>(dev_a, dev_res, ROW, COL);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, start, end);

    cudaMemcpy(res, dev_res, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    if (verify(a, res, ROW, COL) == false) {
        printf("Wrong answer: kernel transpose_rc_wr_1d_shared_buf\n");
        exit(1);
    }

    printf("Success: kernel transpose_rc_wr_1d_shared_buf takes %.3f ms\n", elapsed_time);

    auto grid = dim3((ROW + WARP_SIZE - 1) / WARP_SIZE, (COL + WARP_SIZE - 1) / WARP_SIZE);
    auto block = dim3(WARP_SIZE, WARP_SIZE);

    cudaMemset(dev_res, 0, SIZE * sizeof(int));
    cudaEventRecord(start);
    transpose_rr_wc_2d<<<grid, block>>>(dev_a, dev_res, ROW, COL);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, start, end);

    cudaMemcpy(res, dev_res, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    if (verify(a, res, ROW, COL) == false) {
        printf("Wrong answer: kernel transpose_rr_wc_2d\n");
        exit(1);
    }

    printf("Success: kernel transpose_rr_wc_2d takes %.3f ms\n", elapsed_time);

    cudaMemset(dev_res, 0, SIZE * sizeof(int));
    cudaEventRecord(start);
    transpose_rc_wr_2d<<<grid, block>>>(dev_a, dev_res, ROW, COL);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, start, end);

    cudaMemcpy(res, dev_res, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    if (verify(a, res, ROW, COL) == false) {
        printf("Wrong answer: kernel transpose_rc_wr_2d\n");
        exit(1);
    }

    printf("Success: kernel transpose_rc_wr_2d takes %.3f ms\n", elapsed_time);

    cudaMemset(dev_res, 0, SIZE * sizeof(int));
    cudaEventRecord(start);
    transpose_rr_wc_2d_int4<<<grid, block>>>(dev_a, dev_res, ROW, COL);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, start, end);

    cudaMemcpy(res, dev_res, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    if (verify(a, res, ROW, COL) == false) {
        printf("Wrong answer: kernel transpose_rr_wc_2d_int4\n");
        exit(1);
    }

    printf("Success: kernel transpose_rr_wc_2d_int4 takes %.3f ms\n", elapsed_time);

    cudaMemset(dev_res, 0, SIZE * sizeof(int));
    cudaEventRecord(start);
    transpose_rc_wr_2d_int4<<<grid, block>>>(dev_a, dev_res, ROW, COL);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, start, end);

    cudaMemcpy(res, dev_res, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    if (verify(a, res, ROW, COL) == false) {
        printf("Wrong answer: kernel transpose_rc_wr_2d_int4\n");
        exit(1);
    }

    printf("Success: kernel transpose_rc_wr_2d_int4 takes %.3f ms\n", elapsed_time);

    cudaMemset(dev_res, 0, SIZE * sizeof(int));
    cudaEventRecord(start);
    transpose_rc_wr_2d_int4_shared_buf<<<grid, block>>>(dev_a, dev_res, ROW, COL);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, start, end);

    cudaMemcpy(res, dev_res, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    if (verify(a, res, ROW, COL) == false) {
        printf("Wrong answer: kernel transpose_rc_wr_2d_int4_shared_buf\n");
        exit(1);
    }

    printf("Success: kernel transpose_rc_wr_2d_int4_shared_buf takes %.3f ms\n", elapsed_time);

    cudaFree(dev_a);
    cudaFree(dev_res);
    free(a);
    free(res);

    return 0;
}