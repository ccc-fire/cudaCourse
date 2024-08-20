#include <iostream>
#include <cuda.h>
#include "cuda_runtime.h"

/**
 * reduce_sum: 
 * v0: shared memory -- allcated 100000 blocks, data counts are 25600000
 * latency: 0.992672 ms
 */

template<int blockSize>  // 这是一个模板参数，这个模板参数是一个编译时常量，类型为 int
// 把 blockSize 作为模板参数，能够允许编译时生成多个基于不同 blockSize 值的函数或类的实例
// blockSize作为模板参数的效果主要用于静态shared memory的申请需要传入编译期常量指定大小（L10)
__global__ void reduce_v0(float *d_in, float *d_out) {
    int tid = threadIdx.x;  // 当前 block 内的 thread id
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    // 1D: 当前 grid 所有 block 内的全局 thread id

    __shared__ float smem[blockSize];
    smem[tid] = d_in[gtid];
    __syncthreads();  // 涉及到对shared memory的读写最好都加上__syncthreads

    for(int index = 1; index < blockDim.x; index *= 2) {
        if(tid % (2 * index) == 0) {
            smem[tid] += smem[tid + index];
        }
        __syncthreads();
    }

    // store: 哪里来回哪里去，把reduce结果写回显存
    if(tid == 0) {
        d_out[blockIdx.x] = smem[0];
    }

}

bool checkResult(float *out, float groudtruth, int n) {
    float res = 0;
    for(int i = 0; i < n; i++) {
        res += out[i];
    }
    if(res != groudtruth) {
        return false;
    }
    return true;
}

int main() {
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    const int N = 25600000;
    // const int N = 2.56e+7;  // 容易产生精度问题
    const int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    // int gridSize = std::min((N + 256 - 1) / 256, deviceProp.maxGridSize[0]);

    float *a = (float*)malloc(N * sizeof(float));
    float *d_a;
    cudaMalloc((void **)&d_a, N * sizeof(float));

    float *out = (float*)malloc(gridSize * sizeof(float));
    float *d_out;
    cudaMalloc((void **)&d_out, gridSize * sizeof(float));

    for(int i = 0; i < N; i++) {
        a[i] = 1.0f;
    }

    float groudtruth = N * 1.0f;

    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Grid(gridSize);
    dim3 Block(blockSize);

    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    reduce_v0<blockSize><<<Grid, Block>>>(d_a, d_out);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(out, d_out, gridSize * sizeof(float), cudaMemcpyDeviceToHost);
    printf("allcated %d blocks, data counts are %d\n", gridSize, N);
    bool is_right = checkResult(out, groudtruth, gridSize);
    if(is_right) {
        printf("the ans is right!\n");
    }
    else {
        printf("the ans is wrong!\n");
    }

    printf("reduce_v0 latency = %f ms\n", milliseconds);

    cudaFree(d_a);
    cudaFree(d_out);
    free(a);
    free(out);
}