#include "klist_balance.cuh"
// #define write_to_host


// __global__ void reset_count_buffers(
//     int* count_out_s, int* count_out_m,
//     int* count_in_s, int* count_in_m,
//     int num_items)
// {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tid >= num_items) return;

//     count_out_s[tid] = 0;
//     count_out_m[tid] = 0;
//     count_in_s[tid] = 0;
//     count_in_m[tid] = 0;
// }

// Please do not modift the following, they are correct
__device__ int warp_reduce_sum_balance_buffer(int val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void kmax_scan_balance_buffer(int* t_in_deg, int num_vtx, int* global_buffer, int* buf_count, int level){

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

__global__ void kmax_update_balance_buffer(int* global_buffer, int* buf_count, int* global_count, int* t_in_deg, int* t_out_deg, int* out_offset, int *out_adj, int level){
    
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

__global__ void b_kstatus_update_balance_buffer(int* t_in_deg, bool* kstatus, int num_vtx){

    int tid = blockDim.x * blockIdx.x + threadIdx.x; 
    for(int v = tid; v < num_vtx; v += BLK_DIM * BLK_NUMS){
        kstatus[t_in_deg[v]] = true;
    }

}

__global__ void scan_phase_balance_buffer(int* t_in_deg, int *t_out_deg, int* visit, int num_vtx, int* global_buffer, int* buf_count, int k, int l, int* core){

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

__global__ void update_phase_balance_buffer(int* global_buffer, int* buf_count, int* global_count, int* t_in_deg, int* in_adj, int* in_offset, int* t_out_deg, int* out_adj, int* out_offset, int* visit, int k, int l, int* core){
        
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
// Please do not modify the above, they are correct

__global__ void update_visit_by_core0_balance_buffer(int* core0, int* visit, int num_vtx, int k, int* core){
     
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for(int v = tid; v < num_vtx; v += BLK_NUMS * BLK_DIM){
        visit[v] = (core0[v] >= k);
        if(core0[v] < k){
            core[v] = 0;
            // in_count_num[v] = INT_MAX;
        }
    }
}


__global__ void vertex_to_buffer_buffer(int num_vtx, int* global_buffer, int* buf_count, int* visit){

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


__global__ void vertex_to_buffer_by_out_degree_buffer(int* visit, int num_vtx, int k, int* core, int* out_degree, 
    int* out_buffer_s, int* out_buffer_m, int* out_buffer_l, 
    int* count_out_s, int* count_out_m, int* count_out_l){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int sh_buf_count_s;
    __shared__ int* t_out_buffer_s;
    __shared__ int sh_buf_count_m;
    __shared__ int* t_out_buffer_m;


    if(threadIdx.x == 0){
        sh_buf_count_s = 0;
        sh_buf_count_m = 0;

        t_out_buffer_s = out_buffer_s + blockIdx.x * BUFFER_SIZE;
        t_out_buffer_m = out_buffer_m + blockIdx.x * BUFFER_SIZE;
    }
    __syncthreads();

    for(int v = tid; v < num_vtx; v += BLK_DIM * BLK_NUMS){
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


__global__ void vertex_to_buffer_by_in_degree_buffer(int* visit, int num_vtx, int k, int* core, int* in_degree, 
    int* in_buffer_s, int* in_buffer_m, int* in_buffer_l, 
    int* count_in_s, int* count_in_m, int* count_in_l){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int sh_in_count_s;
    __shared__ int* t_in_buffer_s;
    __shared__ int sh_in_count_m;
    __shared__ int* t_in_buffer_m;


    if(threadIdx.x == 0){
        sh_in_count_s = 0;
        sh_in_count_m = 0;

        t_in_buffer_s = in_buffer_s + blockIdx.x * BUFFER_SIZE;
        t_in_buffer_m = in_buffer_m + blockIdx.x * BUFFER_SIZE;
    }
    __syncthreads();

    for(int v = tid; v < num_vtx; v += BLK_DIM * BLK_NUMS){
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




__global__ void hout_calculate_thread_buffer(int* out_buffer_s, int* count_out_s, int* upper, int* core0, int* hindex_out, int* out_adj, int* out_offset, int k){

    __shared__ int start, end;
    __shared__ int* t_global_buffer;


     if(threadIdx.x == 0){
        t_global_buffer = out_buffer_s + blockIdx.x * BUFFER_SIZE;
        start = 0;
        end = count_out_s[blockIdx.x]; // The end position of the buffer
        assert(t_global_buffer!=NULL);
    } 
    __syncthreads();

    for(int vid = threadIdx.x; vid < end; vid += BLK_DIM){
        
        int v = t_global_buffer[vid];
        int offset_start = out_offset[v]; // offset of v
        int offset_end = out_offset[v+1]; // offset of v

        int local_count = 0;
        int res = upper[v];
        int flag = true;
        while(flag){
            local_count = 0;
            for(int uid = offset_start; uid < offset_end; uid ++){
                int u = out_adj[uid];
                local_count += (core0[u] >= k && upper[u] >= res);
            }
            if(local_count >= res){ hindex_out[v] = res; flag = false;}
            else{
                res -= 1;
            }
        }
        if (res < 0) hindex_out[v] = 0;
    }

}

__global__ void hin_calculate_thread_buffer(int* in_buffer_s, int* count_in_s, int* upper, int* core0, int* hindex_in, int* in_adj, int* in_offset, int k, int* hindex_out){

    __shared__ int start, end;
    __shared__ int* t_global_buffer;


     if(threadIdx.x == 0){
        t_global_buffer = in_buffer_s + blockIdx.x * BUFFER_SIZE;
        start = 0;
        end = count_in_s[blockIdx.x]; // The end position of the buffer
        assert(t_global_buffer!=NULL);
    } 
    __syncthreads();

    for(int vid = threadIdx.x; vid < end; vid += BLK_DIM){
        
        int v = t_global_buffer[vid];
        int offset_start = in_offset[v]; // offset of v
        int offset_end = in_offset[v+1]; // offset of v

        int local_count = 0;
        int res = hindex_out[v];
        int flag = true;

        while(flag){
            local_count = 0;
            for(int uid = offset_start; uid < offset_end; uid ++){
                int u = in_adj[uid];
                local_count += (core0[u] >= k && upper[u] >= res);
            }
            if(local_count >= k){ hindex_in[v] = res; flag = false;}
            else{
                res -= 1;
            }
        }
    }

}

__global__ void hout_calculate_warp_buffer(int* out_buffer_m, int* count_out_m, int* upper, int* core0, int* hindex_out, int* out_adj, int* out_offset, int k){
  
    __shared__ int start, end;
    __shared__ int* t_global_buffer;
    __shared__ int best_mid[BLK_DIM/32];
    __shared__ bool flag[BLK_DIM/32];
    

    int warp_per_block = blockDim.x / WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int start_prime, end_prime;
    if(threadIdx.x == 0){
        t_global_buffer = out_buffer_m + blockIdx.x * BUFFER_SIZE;
        start = 0;
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

        int offset_start = out_offset[v]; // offset of v
        int offset_end = out_offset[v+1]; // offset of v

        if(lane_id == 0){
            best_mid[warp_id] = upper[v];
            flag[warp_id] = true;
        }

        __syncwarp();
        
            
        while(true){
            __syncwarp();
            if(flag[warp_id] == false || best_mid[warp_id] == 0) break;
            __syncwarp();
            int mid = best_mid[warp_id];
            int local_count = 0;

            for(int uid = offset_start+lane_id; uid < offset_end; uid += WARP_SIZE){
                int u = out_adj[uid];
                local_count += (core0[u] >= k && upper[u] >= mid);
            }

            __syncwarp();

            int warp_total = warp_reduce_sum_balance_buffer(local_count);


            if(lane_id == 0){
                if(warp_total >= mid){
                    flag[warp_id] = false;
                }else{
                    best_mid[warp_id] -= 1;
                }
            }
        }
        if(lane_id == 0){
            hindex_out[v] = best_mid[warp_id];
        }
    }
    
}

__global__ void hin_calculate_warp_buffer(int* in_buffer_m, int* count_in_m, int* upper, int* core0, int* hindex_in, int* in_adj, int* in_offset, int k, int* hindex_out){
   
    __shared__ int start, end;
    __shared__ int* t_global_buffer;
    __shared__ int best_mid[BLK_DIM/32];
    __shared__ bool flag[BLK_DIM/32];
    

    int warp_per_block = blockDim.x / WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int start_prime, end_prime;
    if(threadIdx.x == 0){
        t_global_buffer = in_buffer_m + blockIdx.x * BUFFER_SIZE;
        start = 0;
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
        if(hindex_out[v] == 0) continue;    

        int offset_start = in_offset[v]; // offset of v
        int offset_end = in_offset[v+1]; // offset of v

        if(lane_id == 0){
            best_mid[warp_id] = hindex_out[v];
            flag[warp_id] = true;
        }

        __syncwarp();
        
            
        while(true){
            __syncwarp();
            if(flag[warp_id] == false || best_mid[warp_id] == 0) break;
            __syncwarp();
            int mid = best_mid[warp_id];
            int local_count = 0;
        
            for(int uid = offset_start+lane_id; uid < offset_end; uid += WARP_SIZE){
                int u = in_adj[uid];
                local_count += (core0[u] >= k && upper[u] >= mid);
            }
            __syncwarp();
        
            int warp_total = warp_reduce_sum_balance_buffer(local_count);
            if(lane_id == 0){
                if(warp_total >= k){
                    flag[warp_id] = false;
                }else{
                    best_mid[warp_id] -= 1;
                }
            }
        }
        if(lane_id == 0){
            hindex_in[v] = best_mid[warp_id];
        }
    }
    
}

__global__ void hout_calculate_block_buffer(int* out_buffer_l, int* count_out_l, int* upper, int* core0, int* hindex_out, int* out_adj, int* out_offset, int k){
  
    __shared__ int warp_counts[BLK_DIM/WARP_SIZE]; 
    __shared__ int final_count; 
    __shared__ int res_shared;


    int warp_per_block = blockDim.x / WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int tid = threadIdx.x;

    int total = count_out_l[0];



    for(int vid = blockIdx.x; vid < total; vid += gridDim.x){
        int v = out_buffer_l[vid]; // Get the vertex id

        int offset_start = out_offset[v]; // offset of v
        int offset_end = out_offset[v+1]; // offset of v

        int res = upper[v];

        while(res >= 0){
            int local_count = 0;
            for(int uid = offset_start + threadIdx.x; uid < offset_end; uid += blockDim.x){
                int u = out_adj[uid];
                local_count += (core0[u] >= k && upper[u] >= res);
            }

            int warp_sum = warp_reduce_sum_balance_buffer(local_count);
            if (lane_id == 0) warp_counts[warp_id] = warp_sum;
            __syncthreads(); 

            if(tid == 0){
                
                final_count = 0;
                for (int i = 0; i < warp_per_block; i++) {
                    final_count += warp_counts[i];
                }
                if (final_count >= res) {
                    hindex_out[v] = res;
                    res_shared = -1;  // 退出标志
                } else {
                    res_shared = res - 1;
                }
            }
            __syncthreads();
            res = res_shared;
            if(res < 0) break;
        }
    }
}

__global__ void hin_calculate_block_buffer(int* in_buffer_l, int* count_in_l, int* upper, int* core0, int* hindex_in, int* in_adj, int* in_offset, int k, int* hindex_out){
  
    __shared__ int start, end;
    __shared__ int* t_global_buffer;
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

        int offset_start = in_offset[v]; // offset of v
        int offset_end = in_offset[v+1]; // offset of v

        int res = hindex_out[v];

        while(res >= 0){
            int local_count = 0;
            for(int uid = offset_start + threadIdx.x; uid < offset_end; uid += blockDim.x){
                int u = in_adj[uid];
                local_count += (core0[u] >= k && upper[u] >= res);
            }

            int warp_sum = warp_reduce_sum_balance_buffer(local_count);
            if (lane_id == 0) warp_counts[warp_id] = warp_sum;
            __syncthreads(); 

            if(tid == 0){
                final_count = 0;
                for (int i = 0; i < warp_per_block; i++) {
                    final_count += warp_counts[i];
                }
                if (final_count >= k) {
                    hindex_in[v] = res;
                    res_shared = -1;  // 退出标志
                } else {
                    res_shared = res - 1;
                }
            }
            __syncthreads();
            res = res_shared;
            if(res < 0) break;
        }
    }
}


__global__ void update_change_status_buffer(int* global_buffer, int* buf_count, int* hindex_in, int* hindex_out, int* upper, int* in_adj, int* in_offset, int* out_adj, int* out_offset, int* change, int* core0, int k, int* global_done){

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
                int minu = min(hindex_in[o_u], hindex_out[o_u]);
                // if( core0[o_u] >= k && upper[v] >= upper[o_u] && upper[o_u] >= minhindex ){
                if( core0[o_u] >= k && minu > minhindex){
                // if(core0[o_u] >= k && upper[v] >= upper[o_u] && upper[o_u] > minhindex){
                    change[o_u] = 1;
                    *global_done = 1;
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
                int minu = min(hindex_in[i_u], hindex_out[i_u]);
                // printf("min u = %d\n", minu);
                if( core0[i_u] >= k && minu > minhindex){
                // if(core0[i_u] >= k && upper[v] >= upper[i_u] && upper[i_u] > minhindex){
                    change[i_u] = 1;
                    *global_done = 1;
                }
            }
        }

    }
}


__global__ void update_upper_by_visit_buffer(int* global_buffer, int* buf_count, int* hindex_in, int* hindex_out, int* core){

    __shared__ int end;
    __shared__ int* t_global_buffer;

    if(threadIdx.x == 0){
        t_global_buffer = global_buffer + blockIdx.x * BUFFER_SIZE;
        end = buf_count[blockIdx.x]; // The end position of the buffer
        assert(t_global_buffer!=NULL);
    } 

    __syncthreads();

    for(int id = threadIdx.x; id < end; id += BLK_DIM){
        int v = t_global_buffer[id];
        // printf("%d, %d\n", hindex_in[v], hindex_out[v]);
        core[v] = min(hindex_in[v], hindex_out[v]);
    }

}


__global__ void check_innb_cpunt_balance_buffer(int* in_count_num, int* core, int* core0, int num_vtx, int* in_offset, int* in_adj, int k){
    

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = thread_id / WARP_SIZE;
    int total_warps = (gridDim.x * blockDim.x) / WARP_SIZE;

    for (int v = warp_id; v < num_vtx; v += total_warps) {
            if (core0[v] < k) {
                if (lane_id == 0) in_count_num[v] = INT_MAX;
                continue;
            }

            int core_v = core[v];
            int start = in_offset[v];
            int end = in_offset[v+1];
            int cnt = 0;
            for (int i = start + lane_id; i < end; i += WARP_SIZE) {
                int u = in_adj[i];
                cnt += (core[u] >= core_v && core0[u] >= k);
            }

            int warp_total = warp_reduce_sum_balance_buffer(cnt);
            if (lane_id == 0) in_count_num[v] = warp_total;
        }

} 


__device__ int warpReduceMin_balance_buffer(int val) {
    // 使用 warp-level shuffle 归约最小值
    for (int offset = 16; offset > 0; offset /= 2)
        val = min(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    return val;
}

__global__ void reduceMinkernel_balance_buffer(int* in_count_num, int* d_min, int num_vtx){
    __shared__ float sharedMin[256/32];  

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int local_tid = threadIdx.x;

    int min_val = (tid < num_vtx) ? in_count_num[tid] : INT_MAX;

    min_val = warpReduceMin_balance_buffer(min_val);
    
    if (local_tid % WARP_SIZE == 0) {
        sharedMin[local_tid / WARP_SIZE] = min_val;
    }

    __syncthreads();

     // 线程 0 进一步归约共享内存中的最小值
    if (local_tid < WARP_SIZE) {
        int blockMin = (local_tid < 256 / WARP_SIZE) ? sharedMin[local_tid] : INT_MAX;
        blockMin = warpReduceMin_balance_buffer(blockMin);
        if (local_tid == 0) {
            atomicMin(d_min, blockMin);  // 用原子操作更新全局最小值
        }
    }
}



__global__ void update_change_status_out_thread(int* out_buffer_s, int* count_out_s, int* hindex_in, int* hindex_out, int* core, int* out_adj, int* out_offset, int* core0, int k, int * global_done, int* visit){
    
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


__global__ void update_change_status_in_thread(int* in_buffer_s, int* count_in_s, int* hindex_in, int* hindex_out, int* core, int* in_adj, int* in_offset, int* core0, int k, int * global_done, int* visit){
    
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


__global__ void update_change_status_out_warp(int* out_buffer_m, int* count_out_m, int* hindex_in, int* hindex_out, int* core, int* out_adj, int* out_offset, int* core0, int k, int * global_done, int* visit){
    
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


__global__ void update_change_status_in_warp(int* in_buffer_m, int* count_in_m, int* hindex_in, int* hindex_out, int* core, int* in_adj, int* in_offset, int* core0, int k, int * global_done, int* visit){
    
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


__global__ void update_change_status_out_block(int* out_buffer_l, int* count_out_l, int* hindex_in, int* hindex_out, int* core, int* out_adj, int* out_offset, int* core0, int k, int * global_done, int* visit){
    
    int warp_per_block = blockDim.x / WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int tid = threadIdx.x;

    int total = count_out_l[0];

    __shared__ int block_has_change;


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


__global__ void update_change_status_in_block(int* in_buffer_l, int* count_in_l, int* hindex_in, int* hindex_out, int* core, int* in_adj, int* in_offset, int* core0, int k, int * global_done, int* visit){
    
    int warp_per_block = blockDim.x / WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int tid = threadIdx.x;

    int total = count_in_l[0];

    __shared__ int block_has_change;


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


__global__ void update_upper_by_out_buffer_s(int* out_buffer_s, int* count_out_s, int* hindex_in, int* hindex_out, int* core){
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

__global__ void update_upper_by_out_buffer_m(int* out_buffer_m, int* count_out_m, int* hindex_in, int* hindex_out, int* core){

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

__global__ void update_upper_by_out_buffer_l(int* out_buffer_l, int* count_out_l, int* hindex_in, int* hindex_out, int* core){
    
    int total = count_out_l[0];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for(int id = tid; id < total; id += BLK_DIM*BLK_NUMS){
        int v = out_buffer_l[id];
        core[v] = min(hindex_in[v], hindex_out[v]);
    }
}



void klist_balance_buffer_de(G_pointers &p){


    int iterationh = 0;
    int iterationk = 0;

    int max_val = INT_MAX;
    int* d_min;
    chkerr(cudaMalloc(&d_min, sizeof(int))); 

    int kmax = 0;
    int count = 0;
    int* global_count = 0;
    chkerr(cudaMalloc(&global_count, sizeof(int)));


    int* global_buffer;
    chkerr(cudaMalloc(&global_buffer, sizeof(int) * BLK_NUMS * BUFFER_SIZE));

    int* buf_count;
    chkerr(cudaMalloc(&buf_count, sizeof(int) * BLK_NUMS));
    cudaMemset(buf_count, 0, sizeof(int) * BLK_NUMS);

    while(count < p.num_vtx){
        cudaMemset(buf_count, 0, sizeof(int) * BLK_NUMS);
        kmax_scan_balance_buffer<<<BLK_NUMS, BLK_DIM>>>(p.t_in_deg, p.num_vtx, global_buffer, buf_count, kmax);
        kmax_update_balance_buffer<<<BLK_NUMS, BLK_DIM>>>(global_buffer, buf_count, global_count, p.t_in_deg, p.t_out_deg, p.out_offset, p.out_adj, kmax); 
        chkerr(cudaMemcpy(&count, global_count, sizeof(int), cudaMemcpyDeviceToHost));
        kmax ++;
    }

    int* core0;
    chkerr(cudaMalloc(&core0, p.num_vtx*sizeof(int)));
    chkerr(cudaMemcpy(core0, p.t_in_deg, sizeof(int)*p.num_vtx, cudaMemcpyDeviceToDevice));    

    bool* kstatus;
    chkerr(cudaMalloc(&kstatus, kmax * sizeof(bool)));
    bool* h_kstatus = new bool[kmax];
    b_kstatus_update_balance_buffer<<<BLK_NUMS, BLK_DIM>>>(p.t_in_deg, kstatus, p.num_vtx);
    chkerr(cudaMemcpy(h_kstatus, kstatus, sizeof(bool)*kmax, cudaMemcpyDeviceToHost));     

    vector<int> h_kstatus_v;
    for(int i = 0; i < kmax; i ++){
        if(h_kstatus[i]){
            h_kstatus_v.push_back(i);
        }
    }

    cudaMemset(global_count, 0, sizeof(int));
    int l = 0;
    count = 0;
    cudaMemset(p.core, 0, p.num_vtx * sizeof(int));
    chkerr(cudaMemcpy(p.t_in_deg, p.in_deg, p.num_vtx * sizeof(int), cudaMemcpyDeviceToDevice));
    chkerr(cudaMemcpy(p.t_out_deg, p.out_deg, p.num_vtx * sizeof(int), cudaMemcpyDeviceToDevice));

    int* hindex_out;
    chkerr(cudaMalloc(&hindex_out, sizeof(int) * p.num_vtx));
    int* hindex_in;
    chkerr(cudaMalloc(&hindex_in, sizeof(int) * p.num_vtx));

    int* change;
    chkerr(cudaMalloc((&change), p.num_vtx*sizeof(int)));  
        
    int* global_done;
    chkerr(cudaMalloc(&global_done, sizeof(int)));  

    cudaMemset(hindex_in, 0, sizeof(int) * p.num_vtx); // 每一个点in的hindex checked
    cudaMemset(hindex_out, 0, sizeof(int) * p.num_vtx); // 每一个点out的hindex  checked 



    int pos = 0;
    int h_kstatus_v_len = h_kstatus_v.size();
    // std::cout << "len h_kstatus_v_len = " << h_kstatus_v_len << std::endl;

    // the following variable for workload balance.
    int* count_out_s;
    int* count_out_m;
    int* count_out_l;
    int* out_buffer_s;
    int* out_buffer_m;
    int* out_buffer_l;
    cudaMalloc(&count_out_s, sizeof(int) * BLK_NUMS);
    cudaMalloc(&count_out_m, sizeof(int) * BLK_NUMS);
    cudaMalloc(&count_out_l, sizeof(int));
    cudaMalloc(&out_buffer_s, sizeof(int) * BUFFER_SIZE * BLK_NUMS);
    cudaMalloc(&out_buffer_m, sizeof(int) * BUFFER_SIZE * BLK_NUMS);
    cudaMalloc(&out_buffer_l, sizeof(int) * p.num_vtx);
  
   
   
    int* count_in_s;
    int* count_in_m;
    int* count_in_l;
    int* in_buffer_s;
    int* in_buffer_m;
    int* in_buffer_l;
    cudaMalloc(&count_in_s, sizeof(int) * BLK_NUMS);
    cudaMalloc(&count_in_m, sizeof(int) * BLK_NUMS);
    cudaMalloc(&count_in_l, sizeof(int));
    cudaMalloc(&in_buffer_s, sizeof(int) * BUFFER_SIZE * BLK_NUMS);
    cudaMalloc(&in_buffer_m, sizeof(int) * BUFFER_SIZE * BLK_NUMS);
    cudaMalloc(&in_buffer_l, sizeof(int) * p.num_vtx);
    
    #ifdef write_to_host
     std::ifstream file("/home/cheng/DCoreGPU/dataset/em/vtx2id.txt");  // 打开文件
    unordered_map<int, int> id2vtx;
    int vtx, id;
    // 逐行读取数据
    while (file >> vtx >> id) {
        id2vtx[id] = vtx;
    }

    int** res = new int*[kmax];
    for(int l = 0; l < kmax; l ++){
        res[l] = new int[p.num_vtx];
    }
    #endif
    
        // while(pos <= 1){
        
    while(pos < h_kstatus_v_len){
        
        int h_min = INT_MAX; 
        int k = h_kstatus_v[pos];
        cudaMemset(p.visit, 0, p.num_vtx * sizeof(int)); // flag = false means has not visited
        cudaMemset(p.in_count_num, -1, p.num_vtx * sizeof(int));

        if(pos == 0){
            cudaMemset(buf_count, 0, sizeof(int) * BLK_NUMS);
            count = 0;
            l = 0;
            while(count < p.num_vtx){
                scan_phase_balance_buffer<<<BLK_NUMS, BLK_DIM>>>(p.t_in_deg, p.t_out_deg, p.visit, p.num_vtx, global_buffer, buf_count, k, l, p.core); // scan to find the invalid vertex
                update_phase_balance_buffer<<<BLK_NUMS, BLK_DIM>>>(global_buffer, buf_count, global_count, p.t_in_deg, p.in_adj, p.in_offset, p.t_out_deg, p.out_adj, p.out_offset, p.visit, k, l, p.core);// peel the invalid vertex
                chkerr(cudaMemcpy(&count, global_count, sizeof(int), cudaMemcpyDeviceToHost));        //     l ++;
                l ++;
                iterationk ++;
            }
        }else if(pos > 0){
            int done = 1;
            update_visit_by_core0_balance_buffer<<<BLK_NUMS, BLK_DIM>>>(core0, p.visit, p.num_vtx, k, p.core); // 这个在while循环外面
            while(done){
                
                iterationh ++;
                // printf("iteration h = %d\n", iterationh);
                cudaMemset(global_done, 0, sizeof(int));  
                // cudaMemset(change, 0, sizeof(int) * p.num_vtx); 
                cudaMemset(buf_count, 0, sizeof(int) * BLK_NUMS); // buf count  checked

                cudaMemset(count_out_s, 0, sizeof(int) * BLK_NUMS); 
                cudaMemset(count_in_s, 0, sizeof(int) * BLK_NUMS);
                
                cudaMemset(count_out_m, 0, sizeof(int) * BLK_NUMS);
                cudaMemset(count_in_m, 0, sizeof(int) * BLK_NUMS);
                 
                cudaMemset(count_out_l, 0, sizeof(int) * 1); 
                cudaMemset(count_in_l, 0, sizeof(int) * 1);

                // vertex_to_buffer_buffer<<<BLK_NUMS, BLK_DIM>>>(p.num_vtx, global_buffer, buf_count, p.visit);

                // out calculation
                vertex_to_buffer_by_out_degree_buffer<<<BLK_NUMS, BLK_DIM>>>(p.visit, p.num_vtx, k, p.core, p.out_deg, out_buffer_s, out_buffer_m, out_buffer_l, count_out_s, count_out_m, count_out_l);
                hout_calculate_thread_buffer<<<BLK_NUMS, BLK_DIM>>>(out_buffer_s, count_out_s, p.core, core0, hindex_out, p.out_adj, p.out_offset, k);
                hout_calculate_warp_buffer<<<BLK_NUMS, BLK_DIM>>>(out_buffer_m, count_out_m, p.core, core0, hindex_out, p.out_adj, p.out_offset, k);
                hout_calculate_block_buffer<<<BLK_NUMS, BLK_DIM>>>(out_buffer_l, count_out_l, p.core, core0, hindex_out, p.out_adj, p.out_offset, k);
                
                // in calculation
                vertex_to_buffer_by_in_degree_buffer<<<BLK_NUMS, BLK_DIM>>>(p.visit, p.num_vtx, k, p.core, p.in_deg, in_buffer_s, in_buffer_m, in_buffer_l, count_in_s, count_in_m, count_in_l);
                hin_calculate_thread_buffer<<<BLK_NUMS, BLK_DIM>>>(in_buffer_s, count_in_s, p.core, core0, hindex_in, p.in_adj, p.in_offset, k, hindex_out);
                hin_calculate_warp_buffer<<<BLK_NUMS, BLK_DIM>>>(in_buffer_m, count_in_m, p.core, core0, hindex_in, p.in_adj, p.in_offset, k, hindex_out);
                hin_calculate_block_buffer<<<BLK_NUMS, BLK_DIM>>>(in_buffer_l, count_in_l, p.core, core0, hindex_in, p.in_adj, p.in_offset, k, hindex_out);
    
                cudaMemset(p.visit, 0, sizeof(int) * p.num_vtx); 

                // update_change_status_buffer<<<BLK_NUMS, BLK_DIM>>>(global_buffer, buf_count, hindex_in, hindex_out, p.core, p.in_adj, p.in_offset, p.out_adj, p.out_offset, change, core0, k, global_done);
                update_change_status_out_thread<<<BLK_NUMS, BLK_DIM>>>(out_buffer_s, count_out_s, hindex_in, hindex_out, p.core, p.out_adj, p.out_offset, core0, k, global_done, p.visit);
                update_change_status_out_warp<<<BLK_NUMS, BLK_DIM>>>(out_buffer_m, count_out_m, hindex_in, hindex_out, p.core, p.out_adj, p.out_offset, core0, k, global_done, p.visit);
                update_change_status_out_block<<<BLK_NUMS, BLK_DIM>>>(out_buffer_l, count_out_l, hindex_in, hindex_out, p.core, p.out_adj, p.out_offset, core0, k, global_done, p.visit);

                update_change_status_in_thread<<<BLK_NUMS, BLK_DIM>>>(in_buffer_s, count_in_s, hindex_in, hindex_out, p.core, p.in_adj, p.in_offset, core0, k, global_done, p.visit);
                update_change_status_in_warp<<<BLK_NUMS, BLK_DIM>>>(in_buffer_m, count_in_m, hindex_in, hindex_out, p.core, p.in_adj, p.in_offset, core0, k, global_done, p.visit);
                update_change_status_in_block<<<BLK_NUMS, BLK_DIM>>>(in_buffer_l, count_in_l, hindex_in, hindex_out, p.core, p.in_adj, p.in_offset, core0, k, global_done, p.visit);
                

                // update_upper_by_visit_buffer<<<BLK_NUMS, BLK_DIM>>>(global_buffer, buf_count, hindex_in, hindex_out, p.core);
                update_upper_by_out_buffer_s<<<BLK_NUMS, BLK_DIM>>>(out_buffer_s, count_out_s, hindex_in, hindex_out, p.core);
                update_upper_by_out_buffer_m<<<BLK_NUMS, BLK_DIM>>>(out_buffer_m, count_out_m, hindex_in, hindex_out, p.core);
                update_upper_by_out_buffer_l<<<BLK_NUMS, BLK_DIM>>>(out_buffer_l, count_out_l, hindex_in, hindex_out, p.core);
          

                // cudaMemcpy(p.visit, change, sizeof(int)*p.num_vtx, cudaMemcpyDeviceToDevice);
                cudaMemcpy(&done, global_done, sizeof(int), cudaMemcpyDeviceToHost);

            }
        }
        
        if(pos + 1 < h_kstatus_v_len && h_kstatus_v[pos+1] != k+1){
            cudaMemcpy(d_min, &max_val, sizeof(int), cudaMemcpyHostToDevice);
            // b_check_innb_count_ps<<<BLK_NUMS, BLK_DIM>>>(p.in_count_num, p.core, core0, p.num_vtx, p.in_offset, p.in_adj, k);

            check_innb_cpunt_balance_buffer<<<1024, 256>>>(p.in_count_num, p.core, core0, p.num_vtx, p.in_offset, p.in_adj, k);
            
            reduceMinkernel_balance_buffer<<< (p.num_vtx+256-1)/256, 256>>>(p.in_count_num, d_min, p.num_vtx);

            cudaMemcpy(&h_min, d_min, sizeof(int), cudaMemcpyDeviceToHost);
            // cout << "k = " << k << ", h_min = " << h_min << endl;

            if(h_min != INT_MAX && pos+1 < h_kstatus_v_len && h_min+1 < h_kstatus_v[pos+1]){
                h_kstatus_v.insert(h_kstatus_v.begin() + pos + 1, h_min+1);
                h_kstatus_v_len ++;
                // cout << h_min+1 << " is inserted into the list " << endl;
            }
        }
        pos ++;
        #ifdef write_to_host
        chkerr(cudaMemcpy(res[k], p.core, p.num_vtx * sizeof(int), cudaMemcpyDeviceToHost));
        #endif

    }
    
    #ifdef write_to_host

    for(int k = 0; k <h_kstatus_v_len; k ++){
        std::ofstream wr("/home/cheng/DCoreGPU/dataset/em/valance-"+std::to_string(k)+"-gpu.txt");

        for(int v = 0; v < p.num_vtx; v ++){
            wr << id2vtx[v] << " " << res[k][v] << std::endl;
        }
    }

    #endif

    cout << "iterationh = " << iterationh << endl;
    cout << "iterationk = " << iterationk << endl;
    cout << "total iteration = " << iterationk + iterationh << endl;


}