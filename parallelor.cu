#include <cuda_runtime.h>
#include <stdlib.h>
#include <iostream>

using namespace std;

#define MAX_THREADS 1024



__global__ void parallelOr(int *visit, int *global_done, int num_vtx) {
    __shared__ int sdata[32];  // 用于存储每个 warp 的 OR 结果
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    // 1. Warp-Level OR
    unsigned mask = __activemask();  // 计算有效线程掩码
    int warpOr = __any_sync(mask, (idx < num_vtx) ? visit[idx] : 0);  // Warp 内部 OR

    // 2. 让 warp 0 的线程存储结果到 shared memory
    if (lane == 0) {
        sdata[warp_id] = warpOr;
    }
    __syncthreads();

    // 3. 由 warp 0 线程归约 block 内的 OR
    if (threadIdx.x < warpSize) {
        int blockOr = (threadIdx.x < (blockDim.x / warpSize)) ? sdata[threadIdx.x] : 0;
        blockOr = __any_sync(mask, blockOr);

        if (threadIdx.x == 0) {
            atomicOr(global_done, blockOr);  // 只由一个线程写入全局内存
        }
    }
}


int main(){
    int N = 999999;
    int* h_a = new int[N];

    for(int i = 0; i < N; i ++){
        h_a[i] = 0;
    }

    h_a[N-2009] = 0;

    int* d_a;

    cudaMalloc(&d_a, N * sizeof(int));
    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);


    int* global_done;
    cudaMalloc(&global_done, sizeof(int));
    cudaMemset(global_done, 0, sizeof(int));  // 是否完成  checked

    parallelOr<<<(N+1024-1)/1024, 1024>>>(d_a, global_done, N);

    int done;

    cudaMemcpy(&done, global_done, sizeof(int), cudaMemcpyDeviceToHost);

    cout << "done = " << done << endl; 


}