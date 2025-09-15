#include "gpuutilis.cuh"

__device__ int warp_reduce_sum_balance_buffer_utils(int val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}


__device__ int warpReduceMin_balance_buffer_utils(int val) {
    // 使用 warp-level shuffle 归约最小值
    for (int offset = 16; offset > 0; offset /= 2)
        val = min(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    return val;
}

__global__ void reduceMinkernel_balance_buffer_utils(int* in_count_num, int* d_min, int num_vtx){
    __shared__ float sharedMin[256/32];  

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int local_tid = threadIdx.x;

    int min_val = (tid < num_vtx) ? in_count_num[tid] : INT_MAX;

    min_val = warpReduceMin_balance_buffer_utils(min_val);
    
    if (local_tid % WARP_SIZE == 0) {
        sharedMin[local_tid / WARP_SIZE] = min_val;
    }

    __syncthreads();

     // 线程 0 进一步归约共享内存中的最小值
    if (local_tid < WARP_SIZE) {
        int blockMin = (local_tid < 256 / WARP_SIZE) ? sharedMin[local_tid] : INT_MAX;
        blockMin = warpReduceMin_balance_buffer_utils(blockMin);
        if (local_tid == 0) {
            atomicMin(d_min, blockMin);  // 用原子操作更新全局最小值
        }
    }
}



__global__ void hin_count_thread_buffer_utils(int* in_count_num, int* in_buffer_s, int* count_in_s, int* core, int* core0, int* in_adj, int* in_offset, int k){

    __shared__ int end;
    __shared__ int* t_global_buffer;


     if(threadIdx.x == 0){
        t_global_buffer = in_buffer_s + blockIdx.x * BUFFER_SIZE;
        end = count_in_s[blockIdx.x]; // The end position of the buffer
        assert(t_global_buffer!=NULL);
    } 
    __syncthreads();

    for(int vid = threadIdx.x; vid < end; vid += BLK_DIM){
        
        int v = t_global_buffer[vid];
        int core_v = core[v];
        int offset_start = in_offset[v]; // offset of v
        int offset_end = in_offset[v+1]; // offset of v

        int local_count = 0;
        int flag = true;

        for(int uid = offset_start; uid < offset_end; uid ++){
            int u = in_adj[uid];
            local_count += (core0[u] >= k && core[u] >= core_v);
        }
        in_count_num[v] = local_count;
    }

}


__global__ void hin_count_warp_buffer_utils(int* in_count_num, int* in_buffer_m, int* count_in_m, int* core, int* core0, int* in_adj, int* in_offset, int k){
    

    int warp_per_block = blockDim.x / WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int start_prime, end_prime;

    __shared__ int start, end;
    __shared__ int* t_global_buffer;

    if(threadIdx.x == 0){
        start = 0;
        t_global_buffer = in_buffer_m + blockIdx.x * BUFFER_SIZE;
        end = count_in_m[blockIdx.x]; // The end position of the buffer
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
        int core_v = core[v];
        int offset_start = in_offset[v]; // offset of v
        int offset_end = in_offset[v+1]; // offset of v
        
            
     
        int local_count = 0;
        
        for(int uid = offset_start+lane_id; uid < offset_end; uid += WARP_SIZE){
            int u = in_adj[uid];
            local_count += (core0[u] >= k && core[u] >= core_v);
        }
        __syncwarp();
        int warp_total = warp_reduce_sum_balance_buffer_utils(local_count);
            
        if(lane_id == 0){
            in_count_num[v] = warp_total;
        }
    }

} 


__global__ void hin_count_block_buffer_utils(int* in_count_num, int* in_buffer_l, int* count_in_l, int* core, int* core0, int* in_adj, int* in_offset, int k){
    

    __shared__ int warp_counts[BLK_DIM/WARP_SIZE]; 
    __shared__ int final_count; 
    __shared__ int res_shared;

    int warp_per_block = blockDim.x / WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int tid = threadIdx.x;

    int total = count_in_l[0];

    for(int vid = blockIdx.x; vid < total; vid += gridDim.x){
        int v = in_buffer_l[vid]; // Get the vertex id
        int core_v = core[v];
        int offset_start = in_offset[v]; // offset of v
        int offset_end = in_offset[v+1]; // offset of v


        int local_count = 0;
        for(int uid = offset_start + threadIdx.x; uid < offset_end; uid += blockDim.x){
            int u = in_adj[uid];
            local_count += (core0[u] >= k && core[u] >= core_v);
        }

        int warp_sum = warp_reduce_sum_balance_buffer_utils(local_count);
        if (lane_id == 0) warp_counts[warp_id] = warp_sum;
        __syncthreads(); 

        if(tid == 0){
            final_count = 0;
            for (int i = 0; i < warp_per_block; i++) {
                final_count += warp_counts[i];
            }
            in_count_num[v] = final_count;
        }
    }

} 


__global__ void vertex_to_buffer_by_core0_buffer_utils(int k, int* core0, int* in_degree, 
    int* in_buffer_s, int* in_buffer_m, int* in_buffer_l, 
    int* count_in_s, int* count_in_m, int* count_in_l, int* global_buffer, int* buf_count){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int sh_in_count_s;
    __shared__ int* t_in_buffer_s;
    __shared__ int sh_in_count_m;
    __shared__ int* t_in_buffer_m;
    
    __shared__ int* t_global_buffer;
    __shared__ int end;
    


    if(threadIdx.x == 0){
        sh_in_count_s = 0;
        sh_in_count_m = 0;
        end = buf_count[blockIdx.x];

        t_in_buffer_s = in_buffer_s + blockIdx.x * BUFFER_SIZE;
        t_in_buffer_m = in_buffer_m + blockIdx.x * BUFFER_SIZE;
        t_global_buffer = global_buffer + blockIdx.x * BUFFER_SIZE;
    }
    __syncthreads();

    for(int vid = threadIdx.x; vid < end; vid += BLK_DIM){
        int v = t_global_buffer[vid];
        int deg = in_degree[v];

        if(deg <= 16){
            int pos = atomicAdd(&sh_in_count_s, 1);
            t_in_buffer_s[pos] = v;
        }else if(deg <= 1024){
            int pos = atomicAdd(&sh_in_count_m, 1);
            t_in_buffer_m[pos] = v;
        }else if(deg > 1024){
            int pos = atomicAdd(count_in_l, 1);
            in_buffer_l[pos] = v;
        }
    }

    __syncthreads();

    if(threadIdx.x == 0){
        count_in_s[blockIdx.x] = sh_in_count_s;
        count_in_m[blockIdx.x] = sh_in_count_m;
        // printf("sh_in_count_s = %d, sh_in_count_m = %d, sh_in_count_l = %d\n", sh_in_count_s, sh_in_count_m, sh_in_count_l);
    }

}


__global__ void update_upper_by_out_buffer_s_utils(int* out_buffer_s, int* count_out_s, int* hindex_in, int* hindex_out, int* core){
    __shared__ int end;
    __shared__ int* t_global_buffer;

    if(threadIdx.x == 0){
        t_global_buffer = out_buffer_s + blockIdx.x * BUFFER_SIZE;
        end = count_out_s[blockIdx.x]; // The end position of the buffer
        assert(t_global_buffer!=NULL);
    } 
    __syncthreads();

    for(int id = threadIdx.x; id < end; id += BLK_DIM){
        int v = t_global_buffer[id];
        // printf("%d, %d\n", hindex_in[v], hindex_out[v]);
        core[v] = min(hindex_in[v], hindex_out[v]);
    }
}

__global__ void update_upper_by_out_buffer_m_utils(int* out_buffer_m, int* count_out_m, int* hindex_in, int* hindex_out, int* core){

    __shared__ int end;
    __shared__ int* t_global_buffer;

    if(threadIdx.x == 0){
        t_global_buffer = out_buffer_m + blockIdx.x * BUFFER_SIZE;
        end = count_out_m[blockIdx.x]; // The end position of the buffer
        assert(t_global_buffer!=NULL);
    } 

    __syncthreads();

    for(int id = threadIdx.x; id < end; id += BLK_DIM){
        int v = t_global_buffer[id];
        // printf("%d, %d\n", hindex_in[v], hindex_out[v]);
        core[v] = min(hindex_in[v], hindex_out[v]);
    }
    
}

__global__ void update_upper_by_out_buffer_l_utils(int* out_buffer_l, int* count_out_l, int* hindex_in, int* hindex_out, int* core){
    
    int total = count_out_l[0];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for(int id = tid; id < total; id += BLK_DIM*BLK_NUMS){
        int v = out_buffer_l[id];
        core[v] = min(hindex_in[v], hindex_out[v]);
    }
}


__global__ void update_change_status_out_thread_utils(int* out_buffer_s, int* count_out_s, int* hindex_in, int* hindex_out, int* core, int* out_adj, int* out_offset, int* core0, int k, int * global_done, int* visit){
    
    __shared__ int end;
    __shared__ int* t_global_buffer;
    __shared__ int block_has_change;
    
     if(threadIdx.x == 0){
        t_global_buffer = out_buffer_s + blockIdx.x * BUFFER_SIZE;
        end = count_out_s[blockIdx.x]; // The end position of the buffer
        block_has_change = 0;
        assert(t_global_buffer!=NULL);
    } 
    __syncthreads();

    for(int vid = threadIdx.x; vid < end; vid += BLK_DIM){
        
        int v = t_global_buffer[vid];
        int minhindex = min(hindex_in[v], hindex_out[v]);
        if(core[v] <= minhindex) continue;
        int offset_start = out_offset[v]; // offset of v
        int offset_end = out_offset[v+1]; // offset of v

        for(int uid = offset_start; uid < offset_end; uid ++){
            int u = out_adj[uid];
            int minu = min(hindex_in[u], hindex_out[u]);
            if (core0[u] >= k && minu > minhindex) {
                visit[u] = 1;
                block_has_change = 1; //to do
                // atomicOr(block_has_change, 1);
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0 && block_has_change) {
        *global_done = 1;
    }
}

__global__ void update_change_status_in_thread_utils(int* in_buffer_s, int* count_in_s, int* hindex_in, int* hindex_out, int* core, int* in_adj, int* in_offset, int* core0, int k, int * global_done, int* visit){
    
    __shared__ int end;
    __shared__ int* t_global_buffer;
    __shared__ int block_has_change;
    
     if(threadIdx.x == 0){
        t_global_buffer = in_buffer_s + blockIdx.x * BUFFER_SIZE;
        end = count_in_s[blockIdx.x]; // The end position of the buffer
        block_has_change = 0;
        assert(t_global_buffer!=NULL);
    } 
    __syncthreads();

    for(int vid = threadIdx.x; vid < end; vid += BLK_DIM){
        
        int v = t_global_buffer[vid];
        int minhindex = min(hindex_in[v], hindex_out[v]);
        if(core[v] <= minhindex) continue;
        int offset_start = in_offset[v]; // offset of v
        int offset_end = in_offset[v+1]; // offset of v

        for(int uid = offset_start; uid < offset_end; uid ++){
            int u = in_adj[uid];
            int minu = min(hindex_in[u], hindex_out[u]);
            if (core0[u] >= k && minu > minhindex) {
                visit[u] = 1;
                block_has_change = 1; //to do
                // atomicOr(block_has_change, 1);
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0 && block_has_change) {
        *global_done = 1;
    }
}


__global__ void update_change_status_out_warp_utils(int* out_buffer_m, int* count_out_m, int* hindex_in, int* hindex_out, int* core, int* out_adj, int* out_offset, int* core0, int k, int * global_done, int* visit){
    
    __shared__ int start, end;
    __shared__ int* t_global_buffer;
    __shared__ int block_has_change;

    int warp_per_block = blockDim.x / WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int start_prime, end_prime;
    if(threadIdx.x == 0){
        t_global_buffer = out_buffer_m + blockIdx.x * BUFFER_SIZE;
        start = 0;
        block_has_change = 0;
        end = count_out_m[blockIdx.x]; // The end position of the buffer
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
         
        while(true){
            __syncwarp();
            if(core[v] <= minhindex) break;
            if(o_offset_start >= o_offset_end) break;
            int o_uid = o_offset_start + lane_id;
            o_offset_start = o_offset_start + WARP_SIZE; // update the offset position, each thread maintain its own offset_start
            if(o_uid < o_offset_end){
                int o_u = out_adj[o_uid];
                int minu = min(hindex_in[o_u], hindex_out[o_u]);
                if( core0[o_u] >= k && minu > minhindex){
                    visit[o_u] = 1;
                    block_has_change = 1;
                }
            }
        }
    }

    __syncthreads();
    if (threadIdx.x == 0 && block_has_change) {
        *global_done = 1;
    }

}


__global__ void update_change_status_in_warp_utils(int* in_buffer_m, int* count_in_m, int* hindex_in, int* hindex_out, int* core, int* in_adj, int* in_offset, int* core0, int k, int * global_done, int* visit){
    
    __shared__ int start, end;
    __shared__ int* t_global_buffer;
    __shared__ int block_has_change;

    int warp_per_block = blockDim.x / WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int start_prime, end_prime;
    if(threadIdx.x == 0){
        t_global_buffer = in_buffer_m + blockIdx.x * BUFFER_SIZE;
        start = 0;
        block_has_change = 0;
        end = count_in_m[blockIdx.x]; // The end position of the buffer
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

        int i_offset_start = in_offset[v]; 
        int i_offset_end = in_offset[v+1]; 
        
        while(true){
            __syncwarp();
            if(core[v] <= minhindex) break;
            if(i_offset_start >= i_offset_end) break;
            int i_uid = i_offset_start + lane_id; 
            i_offset_start = i_offset_start + WARP_SIZE; 
            if(i_uid < i_offset_end){
                int i_u = in_adj[i_uid];
                int minu = min(hindex_in[i_u], hindex_out[i_u]);
                if( core0[i_u] >= k && minu > minhindex){
                    visit[i_u] = 1;
                    block_has_change = 1;
                }
            }
        }
        
    }

    __syncthreads();
    if (threadIdx.x == 0 && block_has_change) {
        *global_done = 1;
    }

}



__global__ void update_change_status_out_block_utils(int* out_buffer_l, int* count_out_l, int* hindex_in, int* hindex_out, int* core, int* out_adj, int* out_offset, int* core0, int k, int * global_done, int* visit){

    int total = count_out_l[0];



    for(int vid = blockIdx.x; vid < total; vid += gridDim.x){
        
        int v = out_buffer_l[vid]; // Get the vertex id
        int minhindex = min(hindex_in[v], hindex_out[v]);
        if(core[v] <= minhindex) continue;

        int offset_start = out_offset[v]; // offset of v
        int offset_end = out_offset[v+1]; // offset of v
    
        for(int uid = offset_start + threadIdx.x; uid < offset_end; uid += blockDim.x){
            int u = out_adj[uid];
            int minu = min(hindex_in[u], hindex_out[u]);
            if (core0[u] >= k && minu > minhindex) {
                visit[u] = 1;
                atomicExch(global_done, 1);
            }
        }
    }
}


__global__ void update_change_status_in_block_utils(int* in_buffer_l, int* count_in_l, int* hindex_in, int* hindex_out, int* core, int* in_adj, int* in_offset, int* core0, int k, int * global_done, int* visit){
    

    int total = count_in_l[0];


    for(int vid = blockIdx.x; vid < total; vid += gridDim.x){
        
        int v = in_buffer_l[vid]; // Get the vertex id
        int minhindex = min(hindex_in[v], hindex_out[v]);
        if(core[v] <= minhindex) continue;

        int offset_start = in_offset[v]; // offset of v
        int offset_end = in_offset[v+1]; // offset of v
    
        for(int uid = offset_start + threadIdx.x; uid < offset_end; uid += blockDim.x){
            int u = in_adj[uid];
            int minu = min(hindex_in[u], hindex_out[u]);
            if (core0[u] >= k && minu > minhindex) {
                visit[u] = 1;
                atomicExch(global_done, 1);
            }
        }
    }
}



__global__ void vertex_to_buffer_by_out_degree_buffer_utils(int* visit, int num_vtx, int k, int* core, int* out_degree, 
    int* out_buffer_s, int* out_buffer_m, int* out_buffer_l, 
    int* count_out_s, int* count_out_m, int* count_out_l, int* global_buffer, int* buf_count){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int sh_buf_count_s;
    __shared__ int* t_out_buffer_s;
    __shared__ int sh_buf_count_m;
    __shared__ int* t_out_buffer_m;

    __shared__ int* t_global_buffer;
    __shared__ int end;
    


    if(threadIdx.x == 0){
        sh_buf_count_s = 0;
        sh_buf_count_m = 0;
        end = buf_count[blockIdx.x];

        t_out_buffer_s = out_buffer_s + blockIdx.x * BUFFER_SIZE;
        t_out_buffer_m = out_buffer_m + blockIdx.x * BUFFER_SIZE;
        t_global_buffer = global_buffer + blockIdx.x * BUFFER_SIZE;
    }
    __syncthreads();

    for(int vid = threadIdx.x; vid < end; vid += BLK_DIM){
        int v = t_global_buffer[vid];
        int deg = out_degree[v];

        if(deg <= 16 && visit[v] == 1){
            int pos = atomicAdd(&sh_buf_count_s, 1);
            t_out_buffer_s[pos] = v;
        }else if(deg <= 1024 && visit[v] == 1){
            int pos = atomicAdd(&sh_buf_count_m, 1);
            t_out_buffer_m[pos] = v;
        }else if(deg > 1024 && visit[v] == 1){
            int pos = atomicAdd(count_out_l, 1);
            out_buffer_l[pos] = v;
        }
    }

    __syncthreads();

    if(threadIdx.x == 0){
        count_out_s[blockIdx.x] = sh_buf_count_s;
        count_out_m[blockIdx.x] = sh_buf_count_m;
    }

}


__global__ void vertex_to_buffer_by_in_degree_buffer_utils(int* visit, int num_vtx, int k, int* core, int* in_degree, 
    int* in_buffer_s, int* in_buffer_m, int* in_buffer_l, 
    int* count_in_s, int* count_in_m, int* count_in_l, int* global_buffer, int* buf_count){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int sh_in_count_s;
    __shared__ int* t_in_buffer_s;
    __shared__ int sh_in_count_m;
    __shared__ int* t_in_buffer_m;
    
    __shared__ int* t_global_buffer;
    __shared__ int end;
    


    if(threadIdx.x == 0){
        sh_in_count_s = 0;
        sh_in_count_m = 0;
        end = buf_count[blockIdx.x];

        t_in_buffer_s = in_buffer_s + blockIdx.x * BUFFER_SIZE;
        t_in_buffer_m = in_buffer_m + blockIdx.x * BUFFER_SIZE;
        t_global_buffer = global_buffer + blockIdx.x * BUFFER_SIZE;
    }
    __syncthreads();

    for(int vid = threadIdx.x; vid < end; vid += BLK_DIM){
        int v = t_global_buffer[vid];
        int deg = in_degree[v];

        if(deg <= 16 && visit[v] == 1){
            int pos = atomicAdd(&sh_in_count_s, 1);
            t_in_buffer_s[pos] = v;
        }else if(deg <= 1024  && visit[v] == 1){
            int pos = atomicAdd(&sh_in_count_m, 1);
            t_in_buffer_m[pos] = v;
        }else if(deg > 1024 && visit[v] == 1){
            int pos = atomicAdd(count_in_l, 1);
            in_buffer_l[pos] = v;
        }
    }

    __syncthreads();

    if(threadIdx.x == 0){
        count_in_s[blockIdx.x] = sh_in_count_s;
        count_in_m[blockIdx.x] = sh_in_count_m;
        // printf("sh_in_count_s = %d, sh_in_count_m = %d, sh_in_count_l = %d\n", sh_in_count_s, sh_in_count_m, sh_in_count_l);
    }

}


__global__ void vertex_to_buffer_buffer_utils(int num_vtx, int* global_buffer, int* buf_count, int* visit){

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



__global__ void update_visit_by_core0_balance_buffer_utils(int* core0, int* visit, int num_vtx, int k, int* core, int* in_count_number){
     
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for(int v = tid; v < num_vtx; v += BLK_NUMS * BLK_DIM){
        visit[v] = (core0[v] >= k);
        if(core0[v] < k){
            core[v] = 0;
            in_count_number[v] = INT_MAX;
        }
    }
}

__device__ int warpReduceMin_utils(int val) {
    // 使用 warp-level shuffle 归约最小值
    for (int offset = 16; offset > 0; offset /= 2)
        val = min(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    return val;
}


__global__ void klistprune_scan_utils(int* t_in_deg, int num_vtx, int* global_buffer, int* buf_count, int level){

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


__global__ void klistprune_update_utils(int* global_buffer, int* buf_count, int* global_count, int* t_in_deg, int* t_out_deg, int* out_offset, int *out_adj, int level){
    
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


__global__ void kstatus_update_utils(int* t_in_deg, bool* kstatus, int num_vtx){

    int tid = blockDim.x * blockIdx.x + threadIdx.x; 
    for(int v = tid; v < num_vtx; v += BLK_DIM * BLK_NUMS){
        kstatus[t_in_deg[v]] = true;
    }
}


__global__ void scan_block_utils(int* t_in_deg, int *t_out_deg, int* visit, int num_vtx, int* global_buffer, int* buf_count, int k, int l, int* core){

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


__global__ void check_innb_count_utils(int* in_count_num, int* core, int* core0, int num_vtx, int* in_offset, int* in_adj, int k){
    
    int tid = blockDim.x * blockIdx.x + threadIdx.x; 
    
    for(int v = tid; v < num_vtx; v += BLK_DIM * BLK_NUMS){
        int i_offset_start = in_offset[v];
        int i_offset_end = in_offset[v+1];
        int cnt = 0;
        int core_v = core[v];  // 预加载 core[v]
        if(core0[v] < k){
            in_count_num[v] = INT_MAX;
            continue;
        }
        for(int uid = i_offset_start; uid < i_offset_end; uid ++){
            int u = in_adj[uid];
            cnt += (core[u] >= core_v && core0[u] >= k);
        }
        in_count_num[v] = cnt;
    }
} 


__global__ void reduceMinkernel_utils(int* in_count_num, int* d_min, int num_vtx){
    __shared__ float sharedMin[256/32];  

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int local_tid = threadIdx.x;

    int min_val = (tid < num_vtx) ? in_count_num[tid] : INT_MAX;

    min_val = warpReduceMin_utils(min_val);
    
    if (local_tid % WARP_SIZE == 0) {
        sharedMin[local_tid / WARP_SIZE] = min_val;
    }

    __syncthreads();

     // 线程 0 进一步归约共享内存中的最小值
    if (local_tid < WARP_SIZE) {
        int blockMin = (local_tid < 256 / WARP_SIZE) ? sharedMin[local_tid] : INT_MAX;
        blockMin = warpReduceMin_utils(blockMin);
        if (local_tid == 0) {
            atomicMin(d_min, blockMin);  // 用原子操作更新全局最小值
        }
    }
}


__global__ void scan_phase_balance_buffer_utils(int* t_in_deg, int *t_out_deg, int* visit, int num_vtx, int* global_buffer, int* buf_count, int k, int l, int* core){

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


__global__ void update_phase_balance_buffer_utils(int* global_buffer, int* buf_count, int* global_count, int* t_in_deg, int* in_adj, int* in_offset, int* t_out_deg, int* out_adj, int* out_offset, int* visit, int k, int l, int* core){
        
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