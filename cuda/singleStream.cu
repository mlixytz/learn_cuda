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
    cudaStream_t stream;
    int *host_a, *host_b, *host_c;
    int *dev_a, *dev_b, *dev_c;
    // 创建计时器
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    // 初始化流
    cudaStreamCreate(&stream);
    // 在GPU端申请存储
    CHECK(cudaMalloc((void **)&dev_a, N * sizeof(int)));
    CHECK(cudaMalloc((void **)&dev_b, N * sizeof(int)));
    CHECK(cudaMalloc((void **)&dev_c, N * sizeof(int)));
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
    for (int i = 0; i < FULL; i += N)
    {
        // 将数据从CPU锁页内存中传输给GPU显存
        cudaMemcpyAsync(dev_a, host_a + i, N * sizeof(int), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(dev_b, host_b + i, N * sizeof(int), cudaMemcpyHostToDevice, stream);
        kernel<<<N / 256, 256, 0, stream>>>(dev_a, dev_b, dev_c);
        // 将数据从GPU显存中传输给CPU内存
        cudaMemcpyAsync(host_c + i, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost, stream);
    }
    cudaStreamSynchronize(stream);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elaspsedTime, start, stop);
    printf("Time cost: %3.1f ms\n", elaspsedTime);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    cudaFreeHost(host_a);
    cudaFreeHost(host_b);
    cudaFreeHost(host_c);
    cudaStreamDestroy(stream);

    return 0;
}