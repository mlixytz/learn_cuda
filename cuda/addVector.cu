#include <stdio.h>

#include <cuda_runtime.h>

__global__ void vectorAdd(const float* A, const float* B, float* C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

 
int main() {

    // 元素总数
    int numElements = 50000;
    size_t size = numElements * sizeof(float);

    // 在host中创建A、B、C
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    // 初始化 A B C
    for (int i=0; i<numElements; ++i) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    // 在GPU中给三个向量申请空间
    float* d_A = NULL;
    float* d_B = NULL;
    float* d_C = NULL;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // 把数据AB从CPU内存当中复制到GPU显存中
    printf("Copy input data from the host memory to device memory\n");
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice);

    // 执行GPU kernel函数
    int threadPerBlock = 256;
    int blockPerGrid = (numElements + threadPerBlock - 1) / threadPerBlock;
    vectorAdd<<<blockPerGrid, threadPerBlock>>>(d_A, d_B, d_C, numElements);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    for(int i=0; i<numElements; ++i) {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
            fprintf(stderr, "Result verfication failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    printf("Test PASSED\n");
    return 0;
}
