#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>

using namespace std;

#define THREADS_PER_BLOCK 256
#define NUM_BLOCKS 256


float cpudotp(float u[], float v[], int n) {
    float partialSum = 0.0;
    for (int i = 0; i < n; ++i) {
        partialSum = u[i] * v[i];
    }
    return partialSum;
}

__global__ void dotp(const float *u, const float *v, float *partialSum, int n) {

    __shared__ float cache[THREADS_PER_BLOCK];
    int cacheIndex = threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float temp = 0.0;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i = i + stride) {
        temp = temp + u[i] * v[i];
    }
    cache[cacheIndex] = temp;
    __syncthreads();

    int i = blockDim.x / 2;
    while (i > 0) {
        if (cacheIndex < i) {
            cache[cacheIndex] = cache[cacheIndex] + cache[cacheIndex + i];
            __syncthreads();
        }
        i /= 2;
    }

    if (cacheIndex == 0) {
        partialSum[blockIdx.x] = cache[cacheIndex];
    }
}

__global__ void add(int *x, int *y, int *z, int n) {
    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i = i + stride) {
        z[i] = x[i] + y[i];
    }
}

int main() {
    //TODO: Create a
    srand(time(nullptr));
    int N = 256; //Number of elements in arrays

    float *h_x, *h_y, *d_x, *d_y, *h_partialSum, *d_partialSum;

    float  cpu_x [256];
    float cpu_y [256];

    //h_x = new float [N];
    //h_y = new float [N];
    //h_partialSum = new float [THREADS_PER_BLOCK];
    h_x = (float *)malloc(N*sizeof(float));
    h_y = (float *)malloc(N*sizeof(float));
    h_partialSum = (float *)malloc(THREADS_PER_BLOCK*sizeof(float));
    

    cudaMalloc((void **)&d_x, N * sizeof(float));
    cudaMalloc((void **)&d_y, N * sizeof(float));
    cudaMalloc((void **)&d_partialSum, THREADS_PER_BLOCK * sizeof(float));

    //h_x = (float*)malloc(N * sizeof(float));
    //h_y = (float*) malloc(N * sizeof(float));
    //h_partialSum = (float*) malloc(N * sizeof(float));

    for (int i = 0; i < N; ++i) {
        h_x[i] = (float)rand() / (float)RAND_MAX;
        cpu_x[i] = h_x[i];
        h_y[i] = (float)rand() / (float)RAND_MAX;
        cpu_y[i] = h_y[i];
    }



    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice);

   // h_partialSum = (float*)malloc(NUM_BLOCKS * sizeof(float));
    // TODO: Set each element to a random num between 0 and 1

    cout << h_x[0] << " + " << h_y[0] << " = " << h_x[0] + h_y[0] << endl;
    dotp<<<THREADS_PER_BLOCK, NUM_BLOCKS>>>(d_x, d_y, d_partialSum, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_partialSum, d_partialSum, NUM_BLOCKS * sizeof(float), cudaMemcpyDeviceToHost);


    float gpuResult = 0.0;
    for (int i = 0; i < NUM_BLOCKS; ++i) {
        gpuResult = gpuResult + h_partialSum[i];
    }
    cout << "Value calculated from the GPU: " << gpuResult << endl;

    cout << "Value calculated from the CPU: " << cpudotp(cpu_x, cpu_y, N) << endl;


    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_partialSum);

    for (int i = 0; i < N; ++i) {

    }

    /*
    delete [] h_x;
    delete [] h_y;
    delete [] h_partialSum;
    */
    free(h_x);
    free(h_y);
    free(h_partialSum);
}
