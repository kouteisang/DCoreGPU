#include <cuda_runtime.h>
#include <stdlib.h>
#include <iostream>

using namespace std;

#define MAX_THREADS 1024

__global__ void suffixSumKernel(int *data, int N, int P) {
    // `P` is next power-of-2 ≥ N (shared memory length for padded scan)
    extern __shared__ int sdata[];           // shared memory buffer
    int tid = threadIdx.x;
    
    // Load input into shared memory in **reversed order** (to compute suffix as prefix)
    for (unsigned int i = tid; i < P; i += blockDim.x) {
        if (i < N) {
            sdata[i] = data[N - 1 - i];      // reverse copy from global to shared
        } else {
            sdata[i] = 0;                    // pad remaining shared memory with 0
        }
    }
    __syncthreads();
    
    // **Up-sweep (reduce) phase** – build partial sums in a balanced binary tree
    for (unsigned int stride = 1; stride < P; stride *= 2) {
        // Each thread processes multiple tree nodes if needed
        for (unsigned int j = tid; j < P / (2 * stride); j += blockDim.x) {
            unsigned int idx = (j + 1) * 2 * stride - 1;
            sdata[idx] += sdata[idx - stride];
        }
        __syncthreads();
    }
    
    // Set the last element (total sum) to 0 to start **exclusive scan** for down-sweep
    if (tid == 0) {
        sdata[P - 1] = 0;
    }
    __syncthreads();
    
    // **Down-sweep phase** – distribute the sums to get the prefix (suffix) results
    for (unsigned int stride = P / 2; stride >= 1; stride /= 2) {
        for (unsigned int j = tid; j < P / (2 * stride); j += blockDim.x) {
            unsigned int idx = (j + 1) * 2 * stride - 1;
            int t = sdata[idx - stride];
            sdata[idx - stride] = sdata[idx];
            sdata[idx] += t;
        }
        __syncthreads();
    }
    
    // **Write results back** to global memory in-place (compute inclusive suffix sum)
    for (unsigned int i = tid; i < N; i += blockDim.x) {
        // Convert exclusive prefix in `sdata` back to inclusive suffix in original array
        data[N - 1 - i] = sdata[i] + data[N - 1 - i];
    }
}


int main(){
    int N = 4;
    int *h_data = new int[N];
    for(int i = 0; i < N; i ++){
        h_data[i] = 0;
    }
    h_data[3] = 3;

    int* d_data;
    cudaMalloc(&d_data, N * sizeof(int));
    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);

    // int threads = (N < MAX_THREADS) ? N : MAX_THREADS;
    int threads = 1024;
    int P = 1;  while (P < N) P <<= 1;               // next power of two
    cout << "p = " << P << endl;
    size_t shmemSize = P * sizeof(int);
    suffixSumKernel<<<1, threads, shmemSize>>>(d_data, N, P);

    int *cuda_res = new int[N];
    cudaMemcpy(cuda_res, d_data, N * sizeof(int), cudaMemcpyDeviceToHost);

    for(int i = N-2; i>= 0; i --){
        h_data[i] += h_data[i+1];
    }

    // for(int i = 0; i < N; i ++){
    //     cout << "i = " << h_data[i] << endl;
    // }

    for(int i = 0; i < N; i ++){
        if(h_data[i] != cuda_res[i]){
            printf("i = %d", i);
            printf("error!");
            break;
        }
        else{
            printf("h_data = %d, cuda_res = %d\n", h_data[i], cuda_res[i]);
        }
    }
}