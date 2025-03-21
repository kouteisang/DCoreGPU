#include "klistprune.cuh"





__global__ void klistprune_scan_level(int* t_in_deg, int num_vtx, int* global_buffer, int* buf_count, int level){

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


__global__ void klistprune_update_level(int* global_buffer, int* buf_count, int* global_count, int* t_in_deg, int* t_out_deg, int* out_offset, int *out_adj, int level){
    
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



__global__ void klistprune_calculate_scan(int* t_in_deg, int *t_out_deg, int* visit, int num_vtx, int* global_buffer, int* buf_count, int k, int l, int* core){

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



__global__ void klistprune_calculate_update(int* global_buffer, int* buf_count, int* global_count, 
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

__global__ void kstatus_update(int* t_in_deg, bool* kstatus, int num_vtx){

    int tid = blockDim.x * blockIdx.x + threadIdx.x; 
    for(int v = tid; v < num_vtx; v += BLK_DIM * BLK_NUMS){
        kstatus[t_in_deg[v]] = true;
    }

}



__global__ void check_innb_count(int* in_count_num, int* core, int* core0, int num_vtx, int* in_offset, int* in_adj, int k){
    
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


__device__ int warpReduceMin(int val) {
    // 使用 warp-level shuffle 归约最小值
    for (int offset = 16; offset > 0; offset /= 2)
        val = min(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    return val;
}

__global__ void reduceMinkernel(int* in_count_num, int* d_min, int num_vtx){
    __shared__ float sharedMin[256/32];  

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int local_tid = threadIdx.x;

    int min_val = (tid < num_vtx) ? in_count_num[tid] : INT_MAX;

    min_val = warpReduceMin(min_val);
    
    if (local_tid % WARP_SIZE == 0) {
        sharedMin[local_tid / WARP_SIZE] = min_val;
    }

    __syncthreads();

     // 线程 0 进一步归约共享内存中的最小值
    if (local_tid < WARP_SIZE) {
        int blockMin = (local_tid < 256 / WARP_SIZE) ? sharedMin[local_tid] : INT_MAX;
        blockMin = warpReduceMin(blockMin);
        if (local_tid == 0) {
            atomicMin(d_min, blockMin);  // 用原子操作更新全局最小值
        }
    }
}


// __global__ void klistprune_scan_by_core0(int* core0, int* visit, int num_vtx, int* global_buffer, int* buf_count, int* core, int k){

//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     __shared__ int sh_buf_count;
//     __shared__ int* t_global_buffer;

//     if(threadIdx.x == 0){
//         sh_buf_count = 0;
//         t_global_buffer = global_buffer + blockIdx.x * BUFFER_SIZE;
//     }
//     __syncthreads();

//    for(int v = tid; v < num_vtx; v += BLK_DIM * BLK_NUMS){
//         if(core0[v] < k){
//             int pos = atomicAdd(&sh_buf_count, 1);
//             t_global_buffer[pos] = v;
//             visit[v] = 1;
//             core[v] = 0;
//         }
//     }

//     __syncthreads();

//     if(threadIdx.x == 0){
//         buf_count[blockIdx.x] = sh_buf_count;
//     }
// }

// __global__ void klistprune_calculate_by_core0(int* global_buffer, int* buf_count, int* global_count, int* t_in_deg, int* in_adj, int* in_offset, int* t_out_deg, int* out_adj, int* out_offset, int* visit){
    
//     __shared__ int start, end;
//     __shared__ int* t_global_buffer;

//     int warp_per_block = blockDim.x / WARP_SIZE;
//     int warp_id = threadIdx.x / WARP_SIZE;
//     int lane_id = threadIdx.x % WARP_SIZE;
//     int start_prime, end_prime;
//     if(threadIdx.x == 0){
//         t_global_buffer = global_buffer + blockIdx.x * BUFFER_SIZE;
//         start = 0;
//         end = buf_count[blockIdx.x]; // The end position of the buffer
//         assert(t_global_buffer!=NULL);
//     } 

//     __syncthreads();

//     while (true){
//         __syncthreads();
//         if(start >= end) break;
//         start_prime = start + warp_id; // Get the vertex id position
//         end_prime = end; // Get the last position of the vertex id
//         __syncthreads();
//         if(start_prime >= end_prime) continue; // The vertex position is larger than the number of valid vertices in the buffer
//         if(threadIdx.x == 0){
//             start = min(start + warp_per_block, end); // update the start position
//         }
//         int v = t_global_buffer[start_prime]; // Get the vertex id

//         int o_offset_start = out_offset[v]; // offset of v
//         int o_offset_end = out_offset[v+1]; // offset of v
//         // int o2_offset_start = o_offset_start; // offset of v
//         // int o2_offset_end = o_offset_end; // offset of v
         

//         int i_offset_start = in_offset[v];
//         int i_offset_end = in_offset[v+1];

    
//         while(true){
//             __syncwarp();
//             if(o_offset_start >= o_offset_end && i_offset_start >= i_offset_end) break;
//             int o_uid = o_offset_start + lane_id;
//             int i_uid = i_offset_start + lane_id; 
//             o_offset_start = o_offset_start + WARP_SIZE; // update the offset position, each thread maintain its own offset_start
//             i_offset_start = i_offset_start + WARP_SIZE; 
//             if(o_uid < o_offset_end){
//                 int o_u = out_adj[o_uid];
//                 if(visit[o_u] == 0){
//                     atomicSub(&t_in_deg[o_u], 1);
//                 }
//             }
//             __syncwarp();
//             if(i_uid < i_offset_end){
//                 int i_u = in_adj[i_uid];
//                 if(visit[i_u] == 0){
//                     atomicSub(&t_out_deg[i_u], 1);
//                 }
//             }
//         }
//     }
    
//     if(threadIdx.x == 0 && end > 0){
//         atomicAdd(global_count, end);
//     }
// }

void klistprune_de(G_pointers &p){

    int level = 0;
    int count = 0;
    int* global_count = 0;
    chkerr(cudaMalloc(&global_count, sizeof(int)));

    // set the min value
    int max_val = INT_MAX;
    int* d_min;
    chkerr(cudaMalloc(&d_min, sizeof(int)));  
    // int h_min;

    int* buf_count;
    chkerr(cudaMalloc(&buf_count, sizeof(int) * BLK_NUMS));
    cudaMemset(buf_count, 0, sizeof(int) * BLK_NUMS);

    int* global_buffer;
    chkerr(cudaMalloc(&global_buffer, sizeof(int) * BLK_NUMS * BUFFER_SIZE));
    
    int* core0;
    chkerr(cudaMalloc(&core0, p.num_vtx*sizeof(int)));
    

    while(count < p.num_vtx){
        cudaMemset(buf_count, 0, sizeof(int) * BLK_NUMS);
        klistprune_scan_level<<<BLK_NUMS, BLK_DIM>>>(p.t_in_deg, p.num_vtx, global_buffer, buf_count, level);
        klistprune_update_level<<<BLK_NUMS, BLK_DIM>>>(global_buffer, buf_count, global_count, p.t_in_deg, p.t_out_deg, p.out_offset, p.out_adj, level); 
        chkerr(cudaMemcpy(&count, global_count, sizeof(int), cudaMemcpyDeviceToHost));
        level ++;
    }
    chkerr(cudaMemcpy(core0, p.t_in_deg, sizeof(int)*p.num_vtx, cudaMemcpyDeviceToDevice));     


    cout << "level = " << level-1 << endl;

    bool* kstatus;
    chkerr(cudaMalloc(&kstatus, level * sizeof(bool)));
    bool* h_kstatus = new bool[level];
    kstatus_update<<<BLK_NUMS, BLK_DIM>>>(p.t_in_deg, kstatus, p.num_vtx);
    chkerr(cudaMemcpy(h_kstatus, kstatus, sizeof(bool)*level, cudaMemcpyDeviceToHost));     

    // int process = 0;
    vector<int> h_kstatus_v;
    for(int i = 0; i < level; i ++){
        if(h_kstatus[i]){
            h_kstatus_v.push_back(i);
            // cout << "i = " << i << endl;
        }
    }

    // if(!h_kstatus[0]){
    //     h_kstatus_v.insert(h_kstatus_v.begin(), 0);
    // }

    int pos = 0;
    int l = 0;
    count = 0;
    int h_kstatus_v_len = h_kstatus_v.size();
    while(pos < h_kstatus_v_len){
        int h_min = INT_MAX; 
        int k = h_kstatus_v[pos];
        cudaMemset(p.in_count_num, -1, p.num_vtx * sizeof(int));
        cudaMemset(p.core, -1, p.num_vtx * sizeof(int));
        chkerr(cudaMemcpy(p.t_in_deg, p.in_deg, p.num_vtx * sizeof(int), cudaMemcpyDeviceToDevice));
        chkerr(cudaMemcpy(p.t_out_deg, p.out_deg, p.num_vtx * sizeof(int), cudaMemcpyDeviceToDevice));
        cudaMemset(p.visit, 0, p.num_vtx * sizeof(int)); // flag = false means has not visited
        cudaMemset(buf_count, 0, sizeof(int) * BLK_NUMS);
        cudaMemset(global_count, 0, sizeof(int));
        count = 0;
        l = 0;

        // klistprune_scan_by_core0<<<BLK_NUMS, BLK_DIM>>>(core0, p.visit, p.num_vtx, global_buffer, buf_count, p.core, k);
        // klistprune_calculate_by_core0<<<BLK_NUMS, BLK_DIM>>>(global_buffer, buf_count, global_count, p.t_in_deg, p.in_adj, p.in_offset, p.t_out_deg, p.out_adj, p.out_offset, p.visit); 
        // chkerr(cudaMemcpy(&count, global_count, sizeof(int), cudaMemcpyDeviceToHost));        //     l ++;

        while(count < p.num_vtx){
            klistprune_calculate_scan<<<BLK_NUMS, BLK_DIM>>>(p.t_in_deg, p.t_out_deg, p.visit, p.num_vtx, global_buffer, buf_count, k, l, p.core); // scan to find the invalid vertex
            klistprune_calculate_update<<<BLK_NUMS, BLK_DIM>>>(global_buffer, buf_count, global_count, p.t_in_deg, p.in_adj, p.in_offset, p.t_out_deg, p.out_adj, p.out_offset, p.visit, k, l, p.core);// peel the invalid vertex
            chkerr(cudaMemcpy(&count, global_count, sizeof(int), cudaMemcpyDeviceToHost));        //     l ++;
            l ++;
        }
        if(pos + 1 < h_kstatus_v_len && h_kstatus_v[pos+1] != k+1){
            cudaMemcpy(d_min, &max_val, sizeof(int), cudaMemcpyHostToDevice);
            check_innb_count<<<BLK_NUMS, BLK_DIM>>>(p.in_count_num, p.core, core0, p.num_vtx, p.in_offset, p.in_adj, k);
            reduceMinkernel<<< (p.num_vtx+256-1)/256, 256>>>(p.in_count_num, d_min, p.num_vtx);
            cudaMemcpy(&h_min, d_min, sizeof(int), cudaMemcpyDeviceToHost);
            // cout << "k = " << k << ", h_min = " << h_min << endl;

            if(h_min != INT_MAX && pos+1 < h_kstatus_v_len && h_min+1 < h_kstatus_v[pos+1]){
                h_kstatus_v.insert(h_kstatus_v.begin() + pos + 1, h_min+1);
                h_kstatus_v_len ++;
                // cout << h_min+1 << " is inserted into the list " << endl;
            }
        }
            // get min k
        // }else if(pos == h_kstatus_v.size()-1 && h_kstatus_v[pos] != level - 1){
        //     cudaMemcpy(d_min, &max_val, sizeof(int), cudaMemcpyHostToDevice);
        //     check_innb_count<<<BLK_NUMS, BLK_DIM>>>(p.in_count_num, p.core, core0, p.num_vtx, p.in_offset, p.in_adj, k);
        //     reduceMinkernel<<< (p.num_vtx+256-1)/256, 256>>>(p.in_count_num, d_min, p.num_vtx);
        //     cudaMemcpy(&h_min, d_min, sizeof(int), cudaMemcpyDeviceToHost);
        //     cout << "h_min = " << h_min << endl;
        //     // get min k
        //     cout << "I am here 313" << endl; 
        // }
        pos ++;
    }

    // Save to local the k_status
    // std::ofstream file("/home/cheng/DCoreGPU/dataset/hollywood-2011/kstatus.txt");  // 打开文件
    // for(int i = 0; i < h_kstatus_v.size(); i ++){
    //     file << h_kstatus_v[i] << std::endl; 
    // }

    // for(int k = 0; k < level; k ++){
    //     std::ofstream wr("/home/cheng/DCoreGPU/dataset/hollywood-2009-20/hollywood-2009-20-k-"+std::to_string(k)+"-gpu.txt");

    //     for(int v = 0; v < p.num_vtx; v ++){
    //         wr << id2vtx[v] << " " << res[k][v] << std::endl;
    //     }
    // }
    cout << h_kstatus_v.size() << endl;
}