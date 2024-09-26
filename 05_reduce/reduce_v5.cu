/**
 * reduce_sum: 循环展开
 * v5: 完全展开for循环，省掉for循环中的判断和加法指令
 * latency: 0.234016 ms
 */

#include <iostream>
#include <cuda.h>
#include "cuda_runtime.h"

#define THREAD_PER_BLOCK 256

__device__ void blockSharedMemReduce(float* smem) {

}

template <int blockSize>
__global__ void reduce_v5(float* d_in, float* d_out) {
    __shared__ float smem[THREAD_PER_BLOCK];
    size_t tid = threadIdx.x;

}

bool check_result(float *out, float groundtruth, int n) {
    float res = 0;
    for(int i = 0; i < n; i++) {
        res += out[i];
    }
    if(res != groundtruth) {
        return false;
    }
    return true;
}


int main() {
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    const int N = 25600000;
    const int blockSize = 256;
    int gridSize = std::min((N + blockSize - 1) / blockSize, deviceProp.maxGridSize[0]);

    float *a = (float*)malloc(N * sizeof(float));
    float *out = (float*)malloc(gridSize * sizeof(float));
    for(int i = 0; i < N; i++) {
        a[i] = 1.0f;
    }
    float groundtruth = N * 1.0f;

    float *d_a, *d_out;
    cudaMalloc((void **)d_a, N * sizeof(float));
    cudaMalloc((void **)d_out, gridSize * sizeof(float));

    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
    dim3 Grid(gridSize);
    dim3 Block(blockSize / 2);
    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    reduce_v5<blockSize / 2><<<Grid, Block>>>(d_a, d_out);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaMemcpy(out, d_out, gridSize * sizeof(float), cudaMemcpyDeviceToHost);

    printf("allcated %d blocks, data counts are %d \n", gridSize, N);
    bool is_right = check_result(out, groundtruth, gridSize);
    if(is_right) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
        for(int i = 0; i < gridSize;i++){
            printf("resPerBlock : %lf ",out[i]);
        }
        printf("\n");
        printf("groudtruth is: %f \n", groundtruth);
    }
    printf("reduce_v5 latency = %f ms\n", milliseconds);

    cudaFree(d_a);
    cudaFree(d_out);
    free(a);
    free(out);

}