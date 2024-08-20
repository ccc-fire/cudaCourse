/**
 * reduce_sum: 基于v2做出改进
 * v3: 让空闲进程也干活
 * latency: 0.234016 ms
 */

#include <iostream>
#include <cuda.h>
#include "cuda_runtime.h"

template<int blockSize>
__global__ void reduce_v3(float* d_in, float* d_out) {
    __shared__ float smem[blockSize];
    size_t tid = threadIdx.x;
    size_t gtid = blockIdx.x * (blockSize * 2) + threadIdx.x;

    // 每个线程加载两个元素到shared memory
    smem[tid] = d_in[gtid] + d_in[gtid + blockSize];
    __syncthreads();

    // 不断地对半相加，以消除bank conflict
    for(int index = blockDim.x / 2; index > 0; index >>= 1) {
        if(tid < index) {
            smem[tid] += smem[tid + index];
        }
        __syncthreads();
    }

    if(tid == 0) {
        d_out[blockIdx.x] = smem[0];
    } 
}

bool checkResult(float* out, float groundtruth, int n) {
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

    float *d_a;
    cudaMalloc((void**)&d_a, N * sizeof(float));
    float *d_out;
    cudaMalloc((void**)&d_out, gridSize * sizeof(float));
    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Grid(gridSize);
    dim3 Block(blockSize / 2);

    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    reduce_v3<blockSize / 2><<<Grid, Block>>>(d_a, d_out);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(out, d_out, gridSize * sizeof(float), cudaMemcpyDeviceToHost);
    printf("allcated %d blocks, data counts are %d\n", gridSize, N);
    
    bool is_right = checkResult(out, groundtruth, gridSize);
    if(is_right) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
        //for(int i = 0; i < GridSize;i++){
            //printf("res per block : %lf ",out[i]);
        //}
        //printf("\n");
        printf("groudtruth is: %f \n", groundtruth);
    }
    printf("reduce_v2 latency = %f ms\n", milliseconds);

    cudaFree(d_a);
    cudaFree(d_out);
    free(a);
    free(out);
}