#include <stdio.h>
#include <math.h>
#include "error.cuh"

#define BLOCK_SIZE 16

// 每个kernel计算结果矩阵中的一个元素
__global__ void gpu_matrix_mult(int *a, int *b, int *c, int m, int n, int k)
{
    // 二维线程格、二维线程块
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if (col < k && row < m)
    {
        for (int i = 0; i < n; i++)
        {
            sum += a[row * n + i] * b[i * k + col];
        }
    }
    c[row * k + col] = sum;
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

    gpu_matrix_mult<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m, n, k);

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