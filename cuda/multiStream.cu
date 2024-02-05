#include <stdio.h>
#include "error.cuh"

// (A+B)/2=C
#define N (1024 * 1024) // 每个流执行数据大小
#define FULL (N * 20)   // 全部数据大小

__global__ void kernel(int *a, int *b, int *c)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N)
    {
        c[idx] = (a[idx] + b[idx]) / 2;
    }
}

int main()
{
    // 查询设备属性
    cudaDeviceProp prop;
    int whichDevice;
    cudaGetDevice(&whichDevice);
    cudaGetDeviceProperties(&prop, whichDevice);
    if (!prop.deviceOverlap)
    {
        printf("Device will not support overlap!");
        return 0;
    }
    // 初始化计时器事件
    cudaEvent_t start, stop;
    float elaspsedTime;
    // 声明流和Buffer指针
    cudaStream_t stream0;
    cudaStream_t stream1;
    int *host_a, *host_b, *host_c;
    int *dev_a0, *dev_b0, *dev_c0;
    int *dev_a1, *dev_b1, *dev_c1;
    // 创建计时器
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    // 初始化流
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);
    // 在GPU端申请存储
    CHECK(cudaMalloc((void **)&dev_a0, N * sizeof(int)));
    CHECK(cudaMalloc((void **)&dev_b0, N * sizeof(int)));
    CHECK(cudaMalloc((void **)&dev_c0, N * sizeof(int)));
    CHECK(cudaMalloc((void **)&dev_a1, N * sizeof(int)));
    CHECK(cudaMalloc((void **)&dev_b1, N * sizeof(int)));
    CHECK(cudaMalloc((void **)&dev_c1, N * sizeof(int)));
    // 在CPU端申请使用锁页内存
    CHECK(cudaHostAlloc((void **)&host_a, FULL * sizeof(int), cudaHostAllocDefault));
    CHECK(cudaHostAlloc((void **)&host_b, FULL * sizeof(int), cudaHostAllocDefault));
    CHECK(cudaHostAlloc((void **)&host_c, FULL * sizeof(int), cudaHostAllocDefault));
    // 初始化向量A，B向量
    for (int i = 0; i < FULL; i++)
    {
        host_a[i] = rand();
        host_b[i] = rand();
    }
    // 开始计算
    cudaEventRecord(start, 0);
    for (int i = 0; i < FULL; i += 2 * N)
    {
        // 将数据从CPU锁页内存中传输给GPU显存
        cudaMemcpyAsync(dev_a0, host_a + i, N * sizeof(int), cudaMemcpyHostToDevice, stream0);
        cudaMemcpyAsync(dev_a1, host_a + i + N, N * sizeof(int), cudaMemcpyHostToDevice, stream1);
        cudaMemcpyAsync(dev_b0, host_b + i, N * sizeof(int), cudaMemcpyHostToDevice, stream0);
        cudaMemcpyAsync(dev_b1, host_b + i + N, N * sizeof(int), cudaMemcpyHostToDevice, stream1);
        kernel<<<N / 256, 256, 0, stream0>>>(dev_a0, dev_b0, dev_c0);
        kernel<<<N / 256, 256, 0, stream1>>>(dev_a1, dev_b1, dev_c1);
        // 将数据从GPU显存中传输给CPU内存
        cudaMemcpyAsync(host_c + i, dev_c0, N * sizeof(int), cudaMemcpyDeviceToHost, stream0);
        cudaMemcpyAsync(host_c + i + N, dev_c1, N * sizeof(int), cudaMemcpyDeviceToHost, stream1);
    }
    cudaStreamSynchronize(stream0);
    cudaStreamSynchronize(stream1);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elaspsedTime, start, stop);
    printf("Time cost: %3.1f ms\n", elaspsedTime);
    cudaFree(dev_a0);
    cudaFree(dev_b0);
    cudaFree(dev_c0);
    cudaFree(dev_a1);
    cudaFree(dev_b1);
    cudaFree(dev_c1);
    cudaFreeHost(host_a);
    cudaFreeHost(host_b);
    cudaFreeHost(host_c);
    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);

    return 0;
}