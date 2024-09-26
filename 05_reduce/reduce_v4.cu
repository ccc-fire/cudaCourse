/**
 * reduce_sum: 基于v3做出改进
 * v4: 把最外层的 warp拿出来做reduce，避免多做一次sync threads
 * latency: 0.179808 ms
 */

#include <iostream>
#include <cuda.h>
#include "cuda_runtime.h"

__device__ void warpSharedMemReduce(volatile float* smem, int tid) {
    // volatile 关键字避免编译器过度优化
    float x = smem[tid];
    if (blockDim.x >= 64) {
        x += smem[tid + 32]; __syncwarp();
        smem[tid] = x; __syncwarp();
    }
    x += smem[tid + 16]; __syncwarp();
    smem[tid] = x;  __syncwarp();
    x += smem[tid + 8]; __syncwarp();
    smem[tid] = x;  __syncwarp();
    x += smem[tid + 4]; __syncwarp();
    smem[tid] = x;  __syncwarp();
    x += smem[tid + 2]; __syncwarp();
    smem[tid] = x;  __syncwarp();
    x += smem[tid + 1]; __syncwarp();
    smem[tid] = x;  __syncwarp();
}

template<int blockSize>  // 模板参数
__global__ void reduce_v4(float* d_in, float* d_out) {
    __shared__ float smem[blockSize];
    size_t tid = threadIdx.x;
    size_t gtid = blockIdx.x * (blockSize * 2) + threadIdx.x;

    // 每个线程加载两个元素到shared memory
    smem[tid] = d_in[gtid] + d_in[gtid + blockSize];
    __syncthreads();

    // 把最外层的 warp拿出来做reduce，避免多做一次sync threads;
    // 此时一个block对d_in这块数据的reduce_sum结果保存在id为0的线程上面
    // 这里的32是因为一个blocka里面一般情况下是有32个bank的
    for(int index = blockDim.x / 2; index > 32; index >>= 1) {
        if(tid < index) {
            smem[tid] += smem[tid + index];
        }
        __syncthreads();
    }

    if(tid < 32) {
        warpSharedMemReduce(smem, tid);
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
    reduce_v4<blockSize / 2><<<Grid, Block>>>(d_a, d_out);
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
        for(int i = 0; i < gridSize; i++){
            printf("res per block : %lf ",out[i]);
        }
        printf("\n");
        printf("groudtruth is: %f \n", groundtruth);
    }
    printf("reduce_v2 latency = %f ms\n", milliseconds);

    cudaFree(d_a);
    cudaFree(d_out);
    free(a);
    free(out);
}