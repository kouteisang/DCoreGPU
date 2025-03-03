#include "klistanchor.cuh"


__global__ void klistanchor_scan_level(int* t_in_deg, int num_vtx, int* global_buffer, int* buf_count, int level){

    // printf("%d\n", p.num_vtx);
    __shared__ int* t_global_buffer;
    __shared__ int sh_buf_count;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(threadIdx.x == 0){
        sh_buf_count = 0;
        t_global_buffer = global_buffer + blockIdx.x * BUFFER_SIZE;
    }
    __syncthreads();

    for(int v = tid; v < num_vtx; v += BLK_DIM * BLK_NUMS){
        if(t_in_deg[v] == level){
            int pos = atomicAdd(&sh_buf_count, 1);
            t_global_buffer[pos] = v;
        }
    }
    __syncthreads();

    if(threadIdx.x == 0){
        buf_count[blockIdx.x] = sh_buf_count;
    
    }
}


__global__ void klistanchor_update_level(int* global_buffer, int* buf_count, int* global_count, int* t_in_deg, int* t_out_deg, int* out_offset, int *out_adj, int level){
    
    __shared__ int start, end;
    __shared__ int* t_global_buffer;

    int warp_per_block = blockDim.x / WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int start_prime, end_prime;
    if(threadIdx.x == 0){
        t_global_buffer = global_buffer + blockIdx.x * BUFFER_SIZE;
        start = 0;
        end = buf_count[blockIdx.x]; // The end position of the buffer
        // printf("id = %d, end = %d\n", blockIdx.x, end);
    } 

    __syncthreads();

    while(true){
        __syncthreads();
        // printf("end = %d\n", end);
        if(start >= end) break; // All the thread break the iteration
        start_prime = start + warp_id; // Get the vertex id position
        end_prime = end; // Get the last position of the vertex id
        __syncthreads();
        if(start_prime >= end_prime) continue; // The vertex position is larger than the number of valid vertices in the buffer
        if(threadIdx.x == 0){
            start = min(start + warp_per_block, end); // update the start position
        }
        int v = t_global_buffer[start_prime]; // Get the vertex id
        int offset_start = out_offset[v]; // offset of v 
        int offset_end = out_offset[v+1]; // offset of v
        
        while (true){
            __syncwarp();
            if(offset_start >= offset_end) break;
            int uid = offset_start + lane_id;
            offset_start = offset_start + WARP_SIZE; // update the offset position, each thread maintain its own offset_start
            if(uid >= offset_end) continue; // This vertex does not has so many neighbouthood
            int u = out_adj[uid]; // v's out-neighbouthood u
            if(t_in_deg[u] > level){
                int in_deg_u = atomicSub(&t_in_deg[u], 1);
                if(in_deg_u == (level+1)){
                    int end_pos = atomicAdd(&end, 1);
                    t_global_buffer[end_pos] = u;
                }
                if(in_deg_u <= level) { // Add it back
                    atomicAdd(&t_in_deg[u], 1);
                }
            }
        }   
    }

    if(threadIdx.x == 0 && end > 0){
        atomicAdd(global_count, end);
    }


}

__global__ void kanchorstatus_update(int* t_in_deg, bool* kstatus, int num_vtx){

    int tid = blockDim.x * blockIdx.x + threadIdx.x; 
    for(int v = tid; v < num_vtx; v += BLK_DIM * BLK_NUMS){
        kstatus[t_in_deg[v]] = true;
    }

}

__global__ void klistanchor_calculate_scan(int* t_in_deg, int *t_out_deg, int* visit, int num_vtx, int* global_buffer, int* buf_count, int k, int l, int* core){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int sh_buf_count;
    __shared__ int* t_global_buffer;

    if(threadIdx.x == 0){
        sh_buf_count = 0;
        t_global_buffer = global_buffer + blockIdx.x * BUFFER_SIZE;
    }
    __syncthreads();

   for(int v = tid; v < num_vtx; v += BLK_DIM * BLK_NUMS){
        if(visit[v] == 0 && t_out_deg[v] == l){
            int pos = atomicAdd(&sh_buf_count, 1);
            t_global_buffer[pos] = v;
            visit[v] = 1;
            core[v] = l;
        }else if(visit[v] == 0 && t_in_deg[v] < k){
            int pos = atomicAdd(&sh_buf_count, 1);
            t_global_buffer[pos] = v;
            t_out_deg[v] = l;
            visit[v] = 1;
            core[v] = l;
        }
    }

    __syncthreads();

    if(threadIdx.x == 0){
        buf_count[blockIdx.x] = sh_buf_count;
    }

}

__global__ void klistanchor_calculate_update(int* global_buffer, int* buf_count, int* global_count, 
                                int* t_in_deg, int* in_adj, int* in_offset, 
                                int* t_out_deg, int* out_adj, int* out_offset, 
                                int* visit, int k, int l, int* core){
        
    __shared__ int start, end;
    __shared__ int* t_global_buffer;

    int warp_per_block = blockDim.x / WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int start_prime, end_prime;
    if(threadIdx.x == 0){
        t_global_buffer = global_buffer + blockIdx.x * BUFFER_SIZE;
        start = 0;
        end = buf_count[blockIdx.x]; // The end position of the buffer
        assert(t_global_buffer!=NULL);
    } 

    __syncthreads();

    while (true){
        __syncthreads();
        if(start >= end) break;
        start_prime = start + warp_id; // Get the vertex id position
        end_prime = end; // Get the last position of the vertex id
        __syncthreads();
        if(start_prime >= end_prime) continue; // The vertex position is larger than the number of valid vertices in the buffer
        if(threadIdx.x == 0){
            start = min(start + warp_per_block, end); // update the start position
        }
        int v = t_global_buffer[start_prime]; // Get the vertex id

        int o_offset_start = out_offset[v]; // offset of v
        int o_offset_end = out_offset[v+1]; // offset of v
        // int o2_offset_start = o_offset_start; // offset of v
        // int o2_offset_end = o_offset_end; // offset of v
         

        int i_offset_start = in_offset[v];
        int i_offset_end = in_offset[v+1];

    
        while(true){
            __syncwarp();
            if(o_offset_start >= o_offset_end && i_offset_start >= i_offset_end) break;
            int o_uid = o_offset_start + lane_id;
            int i_uid = i_offset_start + lane_id; 
            o_offset_start = o_offset_start + WARP_SIZE; // update the offset position, each thread maintain its own offset_start
            i_offset_start = i_offset_start + WARP_SIZE; 
            if(o_uid < o_offset_end){
                int o_u = out_adj[o_uid];
                int in_deg_u = atomicSub(&t_in_deg[o_u], 1);
                if(in_deg_u == k && atomicCAS(&visit[o_u], 0, 1) == 0){
                    int end_pos = atomicAdd(&end, 1);
                    t_global_buffer[end_pos] = o_u;
                    core[o_u] = l;
                }
            }
            __syncwarp();
            if(i_uid < i_offset_end){
                int i_u = in_adj[i_uid];
                int out_deg_u = atomicSub(&t_out_deg[i_u], 1);
                if(out_deg_u == (l+1) && atomicCAS(&visit[i_u], 0, 1) == 0){
                    int end_pos = atomicAdd(&end, 1); 
                    t_global_buffer[end_pos] = i_u;
                    core[i_u] = l;
                }
            }
        }
    }
    
    if(threadIdx.x == 0 && end > 0){
        atomicAdd(global_count, end);
    }

}


// __global__ void anchor_check_innb_count(int* in_count_num, int* core, int* core0, int num_vtx, int* in_offset, int* in_adj, int k){
    
//     int tid = blockDim.x * blockIdx.x + threadIdx.x; 
    
//     for(int v = tid; v < num_vtx; v += BLK_DIM * BLK_NUMS){
//         int i_offset_start = in_offset[v];
//         int i_offset_end = in_offset[v+1];
//         int cnt = 0;
//         int core_v = core[v];  // 预加载 core[v]
//         if(core0[v] < k){
//             in_count_num[v] = INT_MAX;
//             continue;
//         }
//         for(int uid = i_offset_start; uid < i_offset_end; uid ++){
//             int u = in_adj[uid];
//             cnt += (core[u] >= core_v && core0[u] >= k);
//         }
//         in_count_num[v] = cnt;
//     }
// } 


__device__ int anchor_warpReduceMin(int val) {
    // 使用 warp-level shuffle 归约最小值
    for (int offset = 16; offset > 0; offset /= 2)
        val = min(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    return val;
}

__global__ void anchor_reduceMinkernel(int* in_count_num, int* d_min, int num_vtx){
    __shared__ float sharedMin[256/32];  

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int local_tid = threadIdx.x;

    int min_val = (tid < num_vtx) ? in_count_num[tid] : INT_MAX;

    min_val = anchor_warpReduceMin(min_val);
    
    if (local_tid % WARP_SIZE == 0) {
        sharedMin[local_tid / WARP_SIZE] = min_val;
    }

    __syncthreads();

     // 线程 0 进一步归约共享内存中的最小值
    if (local_tid < WARP_SIZE) {
        int blockMin = (local_tid < 256 / WARP_SIZE) ? sharedMin[local_tid] : INT_MAX;
        blockMin = anchor_warpReduceMin(blockMin);
        if (local_tid == 0) {
            atomicMin(d_min, blockMin);  // 用原子操作更新全局最小值
        }
    }


}


__global__ void vertex_to_buffer(int num_vtx, int* global_buffer, int* buf_count, int* visit){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int sh_buf_count;
    __shared__ int* t_global_buffer;

    if(threadIdx.x == 0){
        sh_buf_count = 0;
        t_global_buffer = global_buffer + blockIdx.x * BUFFER_SIZE;
    }
    __syncthreads();

   for(int v = tid; v < num_vtx; v += BLK_DIM * BLK_NUMS){
        if(visit[v] == 1){
            int pos = atomicAdd(&sh_buf_count, 1);
            t_global_buffer[pos] = v;
        }
    }

    __syncthreads();

    if(threadIdx.x == 0){
        buf_count[blockIdx.x] = sh_buf_count;
    }
}



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


__global__ void llistanchor_scan_level(int* t_out_deg, int num_vtx, int* global_buffer, int* buf_count, int l){

    __shared__ int* t_global_buffer;
    __shared__ int sh_buf_count;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(threadIdx.x == 0){
        sh_buf_count = 0;
        t_global_buffer = global_buffer + blockIdx.x * BUFFER_SIZE;
    }
    __syncthreads();

    for(int v = tid; v < num_vtx; v += BLK_DIM * BLK_NUMS){
        if(t_out_deg[v] == l){
            int pos = atomicAdd(&sh_buf_count, 1);
            t_global_buffer[pos] = v;
        }
    }
    __syncthreads();

    if(threadIdx.x == 0){
        buf_count[blockIdx.x] = sh_buf_count;
    
    }
}


__global__ void llistanchor_calculate_level(int* global_buffer, int* buf_count, int* global_count, int* t_out_deg, int* t_in_deg, int* in_offset, int* in_adj, int l){
      __shared__ int start, end;
    __shared__ int* t_global_buffer;

    int warp_per_block = blockDim.x / WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int start_prime, end_prime;
    if(threadIdx.x == 0){
        t_global_buffer = global_buffer + blockIdx.x * BUFFER_SIZE;
        start = 0;
        end = buf_count[blockIdx.x]; // The end position of the buffer
        // printf("id = %d, end = %d\n", blockIdx.x, end);
    } 

    __syncthreads();

    while(true){
        __syncthreads();
        // printf("end = %d\n", end);
        if(start >= end) break; // All the thread break the iteration
        start_prime = start + warp_id; // Get the vertex id position
        end_prime = end; // Get the last position of the vertex id
        __syncthreads();
        if(start_prime >= end_prime) continue; // The vertex position is larger than the number of valid vertices in the buffer
        if(threadIdx.x == 0){
            start = min(start + warp_per_block, end); // update the start position
        }
        int v = t_global_buffer[start_prime]; // Get the vertex id
        int offset_start = in_offset[v]; // offset of v 
        int offset_end = in_offset[v+1]; // offset of v
        
        while (true){
            __syncwarp();
            if(offset_start >= offset_end) break;
            int uid = offset_start + lane_id;
            offset_start = offset_start + WARP_SIZE; // update the offset position, each thread maintain its own offset_start
            if(uid >= offset_end) continue; // This vertex does not has so many neighbouthood
            int u = in_adj[uid]; // v's out-neighbouthood u
            if(t_out_deg[u] > l){
                int out_deg_u = atomicSub(&t_out_deg[u], 1);
                if(out_deg_u == (l+1)){
                    int end_pos = atomicAdd(&end, 1);
                    t_global_buffer[end_pos] = u;
                }
                if(out_deg_u <= l) { // Add it back
                    atomicAdd(&t_out_deg[u], 1);
                }
            }
        }   
    }

    if(threadIdx.x == 0 && end > 0){
        atomicAdd(global_count, end);
    }

}


__global__ void histagram_calculation(int* global_buffer, int* buf_count, int* hist_out, int* hist_in, int* out_adj, int* out_offset, int* in_adj, int* in_offset, int* core0, int* upper, int k, int num_vtx, int lmax){

    __shared__ int start, end;
    __shared__ int* t_global_buffer;

    int warp_per_block = blockDim.x / WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int start_prime, end_prime;
    if(threadIdx.x == 0){
        t_global_buffer = global_buffer + blockIdx.x * BUFFER_SIZE;
        start = 0;
        end = buf_count[blockIdx.x]; // The end position of the buffer
        assert(t_global_buffer!=NULL);
    } 
        
    while (true){

        __syncthreads();
        if(start >= end) break;
        start_prime = start + warp_id; // Get the vertex id position
        end_prime = end; // Get the last position of the vertex id
        __syncthreads();
        if(start_prime >= end_prime) continue; // The vertex position is larger than the number of valid vertices in the buffer
        if(threadIdx.x == 0){
            start = min(start + warp_per_block, end); // update the start position
        }
        int v = t_global_buffer[start_prime]; // Get the vertex id

        int o_offset_start = out_offset[v]; // offset of v
        int o_offset_end = out_offset[v+1]; // offset of v

        int i_offset_start = in_offset[v];
        int i_offset_end = in_offset[v+1];

        while(true){
            __syncwarp();
            if(o_offset_start >= o_offset_end && i_offset_start >= i_offset_end) break;
            int o_uid = o_offset_start + lane_id;
            int i_uid = i_offset_start + lane_id;
            o_offset_start = o_offset_start + WARP_SIZE; // update the offset position, each thread maintain its own offset_start
            i_offset_start = i_offset_start + WARP_SIZE; // update the offset position, each thread maintain its own offset_start 
            if(o_uid < o_offset_end){
                int o_u = out_adj[o_uid];
                if(core0[o_u] >= k){
                    atomicAdd(&hist_out[v*lmax+upper[o_u]], 1);
                }
            }
            __syncwarp();
            if(i_uid < i_offset_end){
                int i_u = in_adj[i_uid];
                if(core0[i_u] >= k){
                    atomicAdd(&hist_in[v*lmax+upper[i_u]], 1);
                }
            } 
        }

    }
}


__global__ void update_visit_by_core0(int* core0, int* visit, int num_vtx, int k, int* core){
     
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for(int v = tid; v < num_vtx; v += BLK_NUMS * BLK_DIM){
        visit[v] = (core0[v] >= k);
        if(core0[v] < k){
            core[v] = 0;
        }
    }
}


// lmax = N;
__global__ void suffixsum_calculate(int* global_buffer, int* buf_count, int* hist_out, int N, int P){

    __shared__ int start, end;
    __shared__ int* t_global_buffer;
    __shared__ int* data;
    extern __shared__ int sdata[];   
    int tid = threadIdx.x;

    if(threadIdx.x == 0){
        t_global_buffer = global_buffer + blockIdx.x * BUFFER_SIZE;
        start = 0;
        end = buf_count[blockIdx.x]; // The end position of the buffer
        assert(t_global_buffer!=NULL);
    } 

    __syncthreads();

    while (true){
        __syncthreads();
        if(start >= end) break;
        if(tid == 0){
            data = hist_out + N * t_global_buffer[start];
            // printf("t_global_buffer[start] = %d\n", t_global_buffer[start]);
        }
        __syncthreads();
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

        __syncthreads();
        if(tid == 0){
            start ++;
        }
    }            
}

__global__ void hindex_out_calculate(int* global_buffer, int* buf_count, int* hist_out, int* hindex_out, int* upper, int lmax){

    __shared__ int start, end;
    __shared__ int* t_global_buffer;

    int warp_per_block = blockDim.x / WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int start_prime, end_prime;
    if(threadIdx.x == 0){
        t_global_buffer = global_buffer + blockIdx.x * BUFFER_SIZE;
        start = 0;
        end = buf_count[blockIdx.x]; // The end position of the buffer
        assert(t_global_buffer!=NULL);
    } 

    __syncthreads();

    while (true){
        __syncthreads();
        if(start >= end) break;
        start_prime = start + warp_id; // Get the vertex id position
        end_prime = end; // Get the last position of the vertex id
        __syncthreads();
        if(start_prime >= end_prime) continue; // The vertex position is larger than the number of valid vertices in the buffer
        if(threadIdx.x == 0){
            start = min(start + warp_per_block, end); // update the start position
        }
        int v = t_global_buffer[start_prime]; // Get the vertex id

        int offset_start = 0;
        int offset_end = upper[v];
        
        while(true){
            __syncwarp();
            if(offset_start > offset_end) break;
            int uid = offset_start + lane_id;
            offset_start = offset_start + WARP_SIZE; // update the offset position, each thread maintain its own offset_start
            if(uid <= offset_end && hist_out[v*lmax+uid] >= uid){
                atomicMax(&hindex_out[v], uid);
            }
        }
    }
}


__global__ void hindex_in_calculate(int* global_buffer, int* buf_count, int* hist_in, int* hindex_in, int* upper, int k, int lmax){

    __shared__ int start, end;
    __shared__ int* t_global_buffer;

    int warp_per_block = blockDim.x / WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int start_prime, end_prime;
    if(threadIdx.x == 0){
        t_global_buffer = global_buffer + blockIdx.x * BUFFER_SIZE;
        start = 0;
        end = buf_count[blockIdx.x]; // The end position of the buffer
        assert(t_global_buffer!=NULL);
    } 

    __syncthreads();

    while (true){
        __syncthreads();
        if(start >= end) break;
        start_prime = start + warp_id; // Get the vertex id position
        end_prime = end; // Get the last position of the vertex id
        __syncthreads();
        if(start_prime >= end_prime) continue; // The vertex position is larger than the number of valid vertices in the buffer
        if(threadIdx.x == 0){
            start = min(start + warp_per_block, end); // update the start position
        }
        int v = t_global_buffer[start_prime]; // Get the vertex id

        int offset_start = 0;
        int offset_end = upper[v];
        
        while(true){
            __syncwarp();
            if(offset_start > offset_end) break;
            int uid = offset_start + lane_id;
            offset_start = offset_start + WARP_SIZE; // update the offset position, each thread maintain its own offset_start
            if(uid <= offset_end && hist_in[v*lmax+uid] >= k){
                atomicMax(&hindex_in[v], uid);
            }
        }
    }

}


__global__ void update_change_status(int* global_buffer, int* buf_count, int* hindex_in, int* hindex_out, int* upper, int* in_adj, int* in_offset, int* out_adj, int* out_offset, int* change, int* core0, int k){

     __shared__ int start, end;
    __shared__ int* t_global_buffer;

    int warp_per_block = blockDim.x / WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int start_prime, end_prime;
    if(threadIdx.x == 0){
        t_global_buffer = global_buffer + blockIdx.x * BUFFER_SIZE;
        start = 0;
        end = buf_count[blockIdx.x]; // The end position of the buffer
        assert(t_global_buffer!=NULL);
    } 

    __syncthreads();

     while (true){
        __syncthreads();
        if(start >= end) break;
        start_prime = start + warp_id; // Get the vertex id position
        end_prime = end; // Get the last position of the vertex id
        __syncthreads();
        if(start_prime >= end_prime) continue; // The vertex position is larger than the number of valid vertices in the buffer
        if(threadIdx.x == 0){
            start = min(start + warp_per_block, end); // update the start position
        }
        int v = t_global_buffer[start_prime]; // Get the vertex id
        // int upperv = upper[v];
        int minhindex = min(hindex_in[v], hindex_out[v]);

        int o_offset_start = out_offset[v]; 
        int o_offset_end = out_offset[v+1]; 
         

        int i_offset_start = in_offset[v];
        int i_offset_end = in_offset[v+1];

    
        while(true){
            __syncwarp();
            if(upper[v] <= minhindex) break;
            if(o_offset_start >= o_offset_end) break;
            int o_uid = o_offset_start + lane_id;
            o_offset_start = o_offset_start + WARP_SIZE; // update the offset position, each thread maintain its own offset_start
            if(o_uid < o_offset_end){
                int o_u = out_adj[o_uid];
                if(core0[o_u] >= k){
                // if(core0[o_u] >= k && upper[v] >= upper[o_u] && upper[o_u] > minhindex){
                    change[o_u] = 1;
                }
            }
        }

        __syncwarp();


        while(true){
            __syncwarp();
            if(upper[v] <= minhindex) break;
            if(i_offset_start >= i_offset_end) break;
            int i_uid = i_offset_start + lane_id; 
            i_offset_start = i_offset_start + WARP_SIZE; 
            if(i_uid < i_offset_end){
                int i_u = in_adj[i_uid];
                if(core0[i_u] >= k){
                // if(core0[i_u] >= k && upper[v] >= upper[i_u] && upper[i_u] > minhindex){
                    change[i_u] = 1;
                }
            }
        }

    }
    

}


__global__ void update_upper_by_visit(int* hindex_in, int* hindex_out, int* upper, int* visit, int num_vtx){

    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    for(int v = tid; v < num_vtx; v += BLK_DIM * BLK_NUMS){
        if(visit[v] == 1){
            upper[v] = min(hindex_in[v], hindex_out[v]);
        }
    }

}


void klistanchor_de(G_pointers &p){


    int kmax = 0;
    int count = 0;
    int* global_count = 0;
    chkerr(cudaMalloc(&global_count, sizeof(int)));

    int* hindex_out;
    chkerr(cudaMalloc(&hindex_out, sizeof(int) * p.num_vtx));
    int* hindex_in;
    chkerr(cudaMalloc(&hindex_in, sizeof(int) * p.num_vtx));

    // set the min value
    int max_val = INT_MAX;
    int* d_min;
    chkerr(cudaMalloc(&d_min, sizeof(int)));  
    
    int* global_done;
    chkerr(cudaMalloc(&global_done, sizeof(int)));  

    int* buf_count;
    chkerr(cudaMalloc(&buf_count, sizeof(int) * BLK_NUMS));
    cudaMemset(buf_count, 0, sizeof(int) * BLK_NUMS);

    int* global_buffer;
    chkerr(cudaMalloc(&global_buffer, sizeof(int) * BLK_NUMS * BUFFER_SIZE));
    
    int* core0;
    chkerr(cudaMalloc(&core0, p.num_vtx*sizeof(int)));

    // this is the upper bound for each iteration
    int* upper;
    chkerr(cudaMalloc((&upper), p.num_vtx*sizeof(int)));

    int* change;
    chkerr(cudaMalloc((&change), p.num_vtx*sizeof(int)));   

    while(count < p.num_vtx){
        cudaMemset(buf_count, 0, sizeof(int) * BLK_NUMS);
        klistanchor_scan_level<<<BLK_NUMS, BLK_DIM>>>(p.t_in_deg, p.num_vtx, global_buffer, buf_count, kmax);
        klistanchor_update_level<<<BLK_NUMS, BLK_DIM>>>(global_buffer, buf_count, global_count, p.t_in_deg, p.t_out_deg, p.out_offset, p.out_adj, kmax); 
        chkerr(cudaMemcpy(&count, global_count, sizeof(int), cudaMemcpyDeviceToHost));
        kmax ++;
    }
    chkerr(cudaMemcpy(core0, p.t_in_deg, sizeof(int)*p.num_vtx, cudaMemcpyDeviceToDevice));     


    cout << "kmax = " << kmax-1 << endl;

    count = 0;
    chkerr(cudaMemcpy(p.t_in_deg, p.in_deg, p.num_vtx * sizeof(int), cudaMemcpyDeviceToDevice));
    chkerr(cudaMemcpy(p.t_out_deg, p.out_deg, p.num_vtx * sizeof(int), cudaMemcpyDeviceToDevice));
    cudaMemset(p.visit, 0, p.num_vtx * sizeof(int)); // flag = false means has not visited
    cudaMemset(global_count, 0, sizeof(int));
    int lmax = 0;
    while(count < p.num_vtx){
        cudaMemset(buf_count, 0, sizeof(int) * BLK_NUMS);
        llistanchor_scan_level<<<BLK_NUMS, BLK_DIM>>>(p.t_out_deg, p.num_vtx, global_buffer, buf_count, lmax);
        llistanchor_calculate_level<<<BLK_NUMS,BLK_DIM>>>(global_buffer, buf_count, global_count, p.t_out_deg, p.t_in_deg, p.in_offset, p.in_adj, lmax);
        chkerr(cudaMemcpy(&count, global_count, sizeof(int), cudaMemcpyDeviceToHost)); 
        lmax ++;
    }

    cout << "lmax = " << lmax-1 << endl;

    int sh = 1;  while (sh < lmax) sh <<= 1;               // next power of two

    // int *h_hist_out = new int[p.num_vtx * lmax];
    // int *h_hist_in = new int[p.num_vtx * lmax];

    int* hist_out;
    chkerr(cudaMalloc(&hist_out, sizeof(int) * lmax * p.num_vtx));
    // cudaMemset(hist_out, 0, sizeof(int) * lmax * p.num_vtx); // 每一个点的out-degree histogram  checked

    int* hist_in;
    chkerr(cudaMalloc(&hist_in, sizeof(int) * lmax * p.num_vtx));
    // cudaMemset(hist_in, 0, sizeof(int) * lmax * p.num_vtx); // 每一个点的in-degree histogram  checked


// The following for the kstatus
    // bool* kstatus;
    // chkerr(cudaMalloc(&kstatus, kmax * sizeof(bool)));
    // bool* h_kstatus = new bool[kmax];
    // kanchorstatus_update<<<BLK_NUMS, BLK_DIM>>>(p.t_in_deg, kstatus, p.num_vtx);
    // chkerr(cudaMemcpy(h_kstatus, kstatus, sizeof(bool)*kmax, cudaMemcpyDeviceToHost));     


   

   
    int pos = 0;
    int l = 0;
    count = 0;
    cudaMemset(p.core, 0, p.num_vtx * sizeof(int));
    chkerr(cudaMemcpy(p.t_in_deg, p.in_deg, p.num_vtx * sizeof(int), cudaMemcpyDeviceToDevice));
    chkerr(cudaMemcpy(p.t_out_deg, p.out_deg, p.num_vtx * sizeof(int), cudaMemcpyDeviceToDevice));

    int** res = new int*[kmax];
    for(int l = 0; l < kmax; l ++){
        res[l] = new int[p.num_vtx];
    }

    for(int k = 0; k < kmax; k ++){
        int h_min = INT_MAX; 
        cudaMemset(p.visit, 0, p.num_vtx * sizeof(int)); // flag = false means has not visited
        cudaMemset(buf_count, 0, sizeof(int) * BLK_NUMS);
        cudaMemset(global_count, 0, sizeof(int));


        if(k == 0){
            count = 0;
            l = 0;
            while(count < p.num_vtx){
                klistanchor_calculate_scan<<<BLK_NUMS, BLK_DIM>>>(p.t_in_deg, p.t_out_deg, p.visit, p.num_vtx, global_buffer, buf_count, k, l, p.core); // scan to find the invalid vertex
                klistanchor_calculate_update<<<BLK_NUMS, BLK_DIM>>>(global_buffer, buf_count, global_count, p.t_in_deg, p.in_adj, p.in_offset, p.t_out_deg, p.out_adj, p.out_offset, p.visit, k, l, p.core);// peel the invalid vertex
                chkerr(cudaMemcpy(&count, global_count, sizeof(int), cudaMemcpyDeviceToHost));        //     l ++;
                l ++;
            }
        }else if(k > 0){
            int done = 1;
            update_visit_by_core0<<<BLK_NUMS, BLK_DIM>>>(core0, p.visit, p.num_vtx, k, p.core); // 这个在while循环外面
            while(done){
                cudaMemset(hindex_in, 0, sizeof(int) * p.num_vtx); // 每一个点in的hindex checked
                cudaMemset(hindex_out, 0, sizeof(int) * p.num_vtx); // 每一个点out的hindex  checked
                cudaMemset(hist_in, 0, sizeof(int) * lmax * p.num_vtx); // 每一个点的in-degree histogram  checked
                cudaMemset(hist_out, 0, sizeof(int) * lmax * p.num_vtx); // 每一个点的out-degree histogram  checked
                cudaMemset(global_done, 0, sizeof(int));  // 是否完成  checked
                cudaMemset(buf_count, 0, sizeof(int) * BLK_NUMS); // buf count  checked
                cudaMemset(change, 0, sizeof(int) * p.num_vtx); // 是否改变  checked
                vertex_to_buffer<<<BLK_NUMS, BLK_DIM>>>(p.num_vtx, global_buffer, buf_count, p.visit); // 将需要改变的放在buffer里面并设置visit = 1 checked
                histagram_calculation<<<BLK_NUMS, BLK_DIM>>>(global_buffer, buf_count, hist_out, hist_in, p.out_adj, p.out_offset, p.in_adj, p.in_offset, core0, p.core, k, p.num_vtx, lmax); // 计算直方图
                size_t shmemSize = sh * sizeof(int);
                suffixsum_calculate<<<BLK_NUMS, BLK_DIM, shmemSize>>>(global_buffer, buf_count, hist_out, lmax, sh); // hist_out 的后缀和
                hindex_out_calculate<<<BLK_NUMS, BLK_DIM>>>(global_buffer, buf_count, hist_out, hindex_out, p.core, lmax); // hindex_out for each vertex
                suffixsum_calculate<<<BLK_NUMS, BLK_DIM, shmemSize>>>(global_buffer, buf_count, hist_in, lmax, sh); // hist_in 的后缀和
                hindex_in_calculate<<<BLK_NUMS, BLK_DIM>>>(global_buffer, buf_count, hist_in, hindex_in, p.core, k, lmax);
                update_change_status<<<BLK_NUMS, BLK_DIM>>>(global_buffer, buf_count, hindex_in, hindex_out, p.core, p.in_adj, p.in_offset, p.out_adj, p.out_offset, change, core0, k);
                update_upper_by_visit<<<BLK_NUMS, BLK_DIM>>>(hindex_in, hindex_out, p.core, p.visit, p.num_vtx);// update p.core by p.visit
                cudaMemcpy(p.visit, change, sizeof(int)*p.num_vtx, cudaMemcpyDeviceToDevice);
                parallelOr<<<(p.num_vtx+1024-1)/1024, 1024>>>(p.visit, global_done, p.num_vtx);
                cudaMemcpy(&done, global_done, sizeof(int), cudaMemcpyDeviceToHost);
            }
        }
        chkerr(cudaMemcpy(res[k], p.core, p.num_vtx * sizeof(int), cudaMemcpyDeviceToHost));
    }


    std::ifstream file("/home/cheng/DCoreGPU/dataset/em/vtx2id.txt");  // 打开文件
    unordered_map<int, int> id2vtx;
    int vtx, id;
    // 逐行读取数据
    while (file >> vtx >> id) {
        id2vtx[id] = vtx;
    }


    for(int k = 0; k < kmax; k ++){
        std::ofstream wr("/home/cheng/DCoreGPU/dataset/em/em-"+std::to_string(k)+"-gpu-a4.txt");

        for(int v = 0; v < p.num_vtx; v ++){
            wr << id2vtx[v] << " " << res[k][v] << std::endl;
        }
    }



}