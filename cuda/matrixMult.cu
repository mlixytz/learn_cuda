#include <stdio.h>

#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void gpu_matrix_mult_shared(int *d_a, int *d_b, int *d_result, int m, int n, int k) {
    __shared__ int tile_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int tile_b[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int tmp = 0;
    int idx;

    for (int sub=0; sub < gridDim.x; ++sub) {
        idx = row * n + sub * BLOCK_SIZE + threadIdx.x;
    }

}

int main() {

}