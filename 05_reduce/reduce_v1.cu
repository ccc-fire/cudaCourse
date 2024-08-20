#include <iostream>
#include <cuda.h>
#include "cuda_runtime.h"

/**
 * reduce_sum:
 * v1: 用位运算替换除余操作
 * latency: 0.573472 ms
 */

// 错误检查宏
#define cudaCheckError() {                                      \
    cudaError_t e = cudaGetLastError();                         \
    if (e != cudaSuccess) {                                     \
        printf("CUDA Error %s:%d: %s\n", __FILE__, __LINE__,    \
                cudaGetErrorString(e));                         \
        exit(EXIT_FAILURE);                                     \
    }                                                           \
}

// // 错误检查函数
// void checkCudaError(const char *msg) {
//     cudaError_t err = cudaGetLastError();
//     if (err != cudaSuccess) {
//         fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
//         exit(EXIT_FAILURE);
//     }
// }
// checkCudaError("Failed to allocate device memory for d_a");
// checkCudaError("Failed to allocate device memory for d_out");
// checkCudaError("Failed to copy data to device");
// checkCudaError("Kernel execution failed");
// checkCudaError("Failed to copy data from device");


template<int blockSize>
__global__ void reduce_v1(float *d_in, float *d_out) {
    int tid = threadIdx.x;
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("gridDim.x: %d, blockDim.x: %d\n", gridDim.x, blockDim.x);

    __shared__ float smem[blockSize];
    smem[tid] = d_in[gtid];
    __syncthreads();

    for(int index = 1; index < blockDim.x; index *= 2) {
        // 使用位运算替代v0中的取余操作
        // 注意这里的优先级的问题，==的优先级是高于&的
        if((tid & (2 * index - 1)) == 0) {
            smem[tid] += smem[tid + index];
        }
        __syncthreads();
    }

    // GridSize个block内部的reduce sum已得出，保存到d_out的每个索引位置
    if(tid == 0) {
        d_out[blockIdx.x] = smem[0];
    }
}

bool checkResult(float *out, float groundtruth, int n) {
    float res = 0;
    for(int i = 0; i < n; i++) {
        res += out[i];
    }
    return fabs(res - groundtruth) < 1e-5;
}

int main() {
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    int N = 25600000;
    const int blockSize = 256;
    int gridSize = std::min((N + blockSize - 1) / blockSize, deviceProp.maxGridSize[0]);

    float *a = (float *)malloc(N * sizeof(float));
    float *d_a;
    cudaMalloc((void **)&d_a, N * sizeof(float));

    float *out = (float *)malloc(gridSize * sizeof(float));
    float *d_out;
    cudaMalloc((void **)&d_out, gridSize * sizeof(float));

    for(int i = 0; i < N; i++) {
        a[i] = 1.0f;
    }

    float groundtruth = N * 1.0f;

    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Grid(gridSize);
    dim3 Block(blockSize);

    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    reduce_v1<blockSize><<<Grid, Block>>>(d_a, d_out);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(out, d_out, gridSize * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckError();
    // checkCudaError("Kernel execution failed");

    printf("allcated %d blocks, data counts are %d\n", gridSize, N);

    bool is_right = checkResult(out, groundtruth, gridSize);
    if(is_right) {
        printf("the ans is right!\n");
    } else {
        printf("the ans is wrong: %g\n", *out);
        printf("groundtruth is: %f \n", groundtruth);
    }
    printf("reduce_v1 latency = %f ms\n", milliseconds);

    cudaFree(d_a);
    cudaFree(d_out);
    free(a);
    free(out);
    return 0;
}