#include <stdio.h>
#include <math.h>
#include "error.cuh"

#define BLOCK_SIZE 16

// 每个kernel计算结果矩阵中的一个元素
// 线程数量 = 输出矩阵元素数量
__global__ void gpu_matrix_mult_shared(int *a, int *b, int *d_result, int n)
{
    __shared__ int tile_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int tile_b[BLOCK_SIZE][BLOCK_SIZE];

    // 当前线程的索引
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int tmp = 0;
    int idx;

    // 把矩阵分块
    for (int sub = 0; sub < gridDim.x; ++sub)
    {
        idx = row * n + sub * BLOCK_SIZE + threadIdx.x;
        tile_a[threadIdx.y][threadIdx.x] = row < n && (sub * BLOCK_SIZE + threadIdx.x) < n ? a[idx] : 0;
        idx = sub * BLOCK_SIZE + threadIdx.y * n + col;
        tile_b[threadIdx.y][threadIdx.x] = col < n && (sub * BLOCK_SIZE + threadIdx.y) < n ? b[idx] : 0;

        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            tmp += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < n && col < n)
    {
        d_result[row * n + col] = tmp;
    }
}

int main(int argc, char const *argv[])
{
    int m = 1000;
    int n = 1000;
    int k = 1000;

    int *h_a, *h_b, *h_c;
    CHECK(cudaMallocHost((void **)&h_a, sizeof(int) * m * n));
    CHECK(cudaMallocHost((void **)&h_b, sizeof(int) * n * k));
    CHECK(cudaMallocHost((void **)&h_c, sizeof(int) * m * k));

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            h_a[i * n + j] = 1;
        }
    }
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            h_b[i * k + j] = 0;
        }
    }

    int *d_a, *d_b, *d_c;
    CHECK(cudaMalloc((void **)&d_a, sizeof(int) * m * n));
    CHECK(cudaMalloc((void **)&d_b, sizeof(int) * n * k));
    CHECK(cudaMalloc((void **)&d_c, sizeof(int) * m * k));

    CHECK(cudaEventRecord(start));

    CHECK(cudaMemcpy(d_a, h_a, sizeof(int) * m * n, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, sizeof(int) * n * k, cudaMemcpyHostToDevice));

    // 二维grid 二维block
    unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    gpu_matrix_mult_shared<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, n);

    CHECK(cudaMemcpy(d_c, h_c, sizeof(int) * m * k, cudaMemcpyDeviceToHost));
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float elapsedTime;
    CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("cost = %g ms.\n", elapsedTime);

    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_c));
    CHECK(cudaFreeHost(h_a));
    CHECK(cudaFreeHost(h_b));
    CHECK(cudaFreeHost(h_c));

    return 0;
}