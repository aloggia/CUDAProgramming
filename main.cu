#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

#define THREADS_PER_BLOCK 256
#define NUM_BLOCKS 256

__global__ void dotp(float *u, float *v, float *partialSum, int n) {

    __shared__ float cache[THREADS_PER_BLOCK];
    int cacheIndex = threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float temp = 0.0;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i = i + stride) {
        temp = temp + u[i] * v[i];
    }

    cache[cacheIndex] = temp;
    __syncthreads();

    cacheIndex = threadIdx.x;
    int i = blockDim.x / 2;
    while (i > 0) {
        if (cacheIndex < i) {
            cache[cacheIndex] = cache[cacheIndex] + cache[cacheIndex + i];
            __syncthreads();
        }
        i = i / 2;
    }
}

__global__ void add(int *x, int *y, int *z, int n) {
    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i = i + stride) {
        z[i] = x[i] + y[i];
    }
}

int main() {
    int N = 1 << 20; //Number of elements in arrays
    float *x, *y, *d_x, *d_y, *partialSum, d_partialSum;

    x = (float*)malloc(N * sizeof(float));
    y = (float*) malloc(N * sizeof(float));

    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));

    partialSum = (float*)malloc(NUM_BLOCKS * sizeof(float));
    // TODO: Set each element to a random num between 0 and 1
    for (int i = 0; i < N; ++i) {
        x[i] = (float) (i + 1);
        y[i] = 1.0 / U[i];
    }

    cudaMemcpy( d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy( d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy( d_partialSum, partialSum, NUM_BLOCKS * sizeof(float), cudaMemcpyHostToDevice);


    dotp<<<THREADS_PER_BLOCK, NUM_BLOCKS>>>(d_x, d_y, d_partialSum, N);
    cudaDeviceSynchronize();
    cudaMemcpy(partialSum, d_partialSum, NUM_BLOCKS*sizeof(float), cudaMemcpyHostToDevice);

    float gpuResult = 0.0;
    for (int i = 0; i < NUM_BLOCKS; ++i) {
        gpuResult = gpuResult + partialSum[i];
    }

}
