#include "klist_balance.cuh"


// Please do not modift the following, they are correct
__global__ void kmax_scan_balance(int* t_in_deg, int num_vtx, int* global_buffer, int* buf_count, int level){

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

__global__ void kmax_update_balance(int* global_buffer, int* buf_count, int* global_count, int* t_in_deg, int* t_out_deg, int* out_offset, int *out_adj, int level){
    
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

__global__ void b_kstatus_update_balance(int* t_in_deg, bool* kstatus, int num_vtx){

    int tid = blockDim.x * blockIdx.x + threadIdx.x; 
    for(int v = tid; v < num_vtx; v += BLK_DIM * BLK_NUMS){
        kstatus[t_in_deg[v]] = true;
    }

}

__global__ void scan_phase_balance(int* t_in_deg, int *t_out_deg, int* visit, int num_vtx, int* global_buffer, int* buf_count, int k, int l, int* core){

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

__global__ void update_phase_balance(int* global_buffer, int* buf_count, int* global_count, int* t_in_deg, int* in_adj, int* in_offset, int* t_out_deg, int* out_adj, int* out_offset, int* visit, int k, int l, int* core){
        
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

__global__ void update_visit_by_core0_balance(int* core0, int* visit, int num_vtx, int k, int* core){
     
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for(int v = tid; v < num_vtx; v += BLK_NUMS * BLK_DIM){
        visit[v] = (core0[v] >= k);
        if(core0[v] < k){
            core[v] = 0;
        }
    }
}


__global__ void vertex_to_buffer_by_out_degree(int* visit, int num_vtx, int k, int* core, int* out_degree, 
    int* out_buffer_s, int* out_buffer_m, int* out_buffer_l, 
    int* count_out_s, int* count_out_m, int* count_out_l){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int sh_buf_count_s;
    __shared__ int* t_out_buffer_s;
    __shared__ int sh_buf_count_m;
    __shared__ int* t_out_buffer_m;
    __shared__ int sh_buf_count_l;
    __shared__ int* t_out_buffer_l;


    if(threadIdx.x == 0){
        sh_buf_count_s = 0;
        sh_buf_count_m = 0;
        sh_buf_count_l = 0;

        t_out_buffer_s = out_buffer_s + blockIdx.x * BUFFER_SIZE;
        t_out_buffer_m = out_buffer_m + blockIdx.x * BUFFER_SIZE;
        t_out_buffer_l = out_buffer_l + blockIdx.x * BUFFER_SIZE;
    }
    __syncthreads();

    for(int v = tid; v < num_vtx; v += BLK_DIM * BLK_NUMS){
        if(visit[v] == 1) continue;
        int deg = out_degree[v];

        if(deg < 16){
            int pos = atomicAdd(&sh_buf_count_s, 1);
            t_out_buffer_s[pos] = v;
        }else if(deg < 1024){
            int pos = atomicAdd(&sh_buf_count_m, 1);
            t_out_buffer_m[pos] = v;
        }else{
            int pos = atomicAdd(&sh_buf_count_l, 1);
            t_out_buffer_l[pos] = v;
        }
    }

    __syncthreads();

    if(threadIdx.x == 0){
        count_out_s[blockIdx.x] = sh_buf_count_s;
        count_out_m[blockIdx.x] = sh_buf_count_m;
        count_out_l[blockIdx.x] = sh_buf_count_l;
    }

}


__global__ void vertex_to_buffer_by_in_degree(int* visit, int num_vtx, int k, int* core, int* in_degree, 
    int* in_buffer_s, int* in_buffer_m, int* in_buffer_l, 
    int* count_in_s, int* count_in_m, int* count_in_l){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int sh_in_count_s;
    __shared__ int* t_in_buffer_s;
    __shared__ int sh_in_count_m;
    __shared__ int* t_in_buffer_m;
    __shared__ int sh_in_count_l;
    __shared__ int* t_in_buffer_l;


    if(threadIdx.x == 0){
        sh_in_count_s = 0;
        sh_in_count_m = 0;
        sh_in_count_l = 0;

        t_in_buffer_s = in_buffer_s + blockIdx.x * BUFFER_SIZE;
        t_in_buffer_m = in_buffer_m + blockIdx.x * BUFFER_SIZE;
        t_in_buffer_l = in_buffer_l + blockIdx.x * BUFFER_SIZE;
    }
    __syncthreads();

    for(int v = tid; v < num_vtx; v += BLK_DIM * BLK_NUMS){
        if(visit[v] == 1) continue;
        int deg = in_degree[v];

        if(deg < 16){
            int pos = atomicAdd(&sh_in_count_s, 1);
            t_in_buffer_s[pos] = v;
        }else if(deg < 1024){
            int pos = atomicAdd(&sh_in_count_m, 1);
            t_in_buffer_m[pos] = v;
        }else{
            int pos = atomicAdd(&sh_in_count_l, 1);
            t_in_buffer_l[pos] = v;
        }
    }

    __syncthreads();

    if(threadIdx.x == 0){
        count_in_s[blockIdx.x] = sh_in_count_s;
        count_in_m[blockIdx.x] = sh_in_count_m;
        count_in_l[blockIdx.x] = sh_in_count_l;
    }

}



void klist_balance_de(G_pointers &p){


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
        kmax_scan_balance<<<BLK_NUMS, BLK_DIM>>>(p.t_in_deg, p.num_vtx, global_buffer, buf_count, kmax);
        kmax_update_balance<<<BLK_NUMS, BLK_DIM>>>(global_buffer, buf_count, global_count, p.t_in_deg, p.t_out_deg, p.out_offset, p.out_adj, kmax); 
        chkerr(cudaMemcpy(&count, global_count, sizeof(int), cudaMemcpyDeviceToHost));
        kmax ++;
    }

    int* core0;
    chkerr(cudaMalloc(&core0, p.num_vtx*sizeof(int)));
    chkerr(cudaMemcpy(core0, p.t_in_deg, sizeof(int)*p.num_vtx, cudaMemcpyDeviceToDevice));    

    bool* kstatus;
    chkerr(cudaMalloc(&kstatus, kmax * sizeof(bool)));
    bool* h_kstatus = new bool[kmax];
    b_kstatus_update_balance<<<BLK_NUMS, BLK_DIM>>>(p.t_in_deg, kstatus, p.num_vtx);
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

    // the following variable for workload balance.
    int* count_out_s;
    int* count_out_m;
    int* count_out_l;
    int* out_buffer_s;
    int* out_buffer_m;
    int* out_buffer_l;
    cudaMalloc(&count_out_s, sizeof(int) * BLK_DIM);
    cudaMalloc(&count_out_m, sizeof(int) * BLK_DIM);
    cudaMalloc(&count_out_l, sizeof(int) * BLK_DIM);
    cudaMalloc(&out_buffer_s, sizeof(int) * BLK_DIM * BLK_NUMS);
    cudaMalloc(&out_buffer_m, sizeof(int) * BLK_DIM * BLK_NUMS);
    cudaMalloc(&out_buffer_l, sizeof(int) * BLK_DIM * BLK_NUMS);
  
   
   
    int* count_in_s;
    int* count_in_m;
    int* count_in_l;
    int* in_buffer_s;
    int* in_buffer_m;
    int* in_buffer_l;
    cudaMalloc(&count_in_s, sizeof(int) * BLK_DIM);
    cudaMalloc(&count_in_m, sizeof(int) * BLK_DIM);
    cudaMalloc(&count_in_l, sizeof(int) * BLK_DIM);
    cudaMalloc(&in_buffer_s, sizeof(int) * BLK_DIM * BLK_NUMS);
    cudaMalloc(&in_buffer_m, sizeof(int) * BLK_DIM * BLK_NUMS);
    cudaMalloc(&in_buffer_l, sizeof(int) * BLK_DIM * BLK_NUMS);
    
    
    
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
                scan_phase_balance<<<BLK_NUMS, BLK_DIM>>>(p.t_in_deg, p.t_out_deg, p.visit, p.num_vtx, global_buffer, buf_count, k, l, p.core); // scan to find the invalid vertex
                update_phase_balance<<<BLK_NUMS, BLK_DIM>>>(global_buffer, buf_count, global_count, p.t_in_deg, p.in_adj, p.in_offset, p.t_out_deg, p.out_adj, p.out_offset, p.visit, k, l, p.core);// peel the invalid vertex
                chkerr(cudaMemcpy(&count, global_count, sizeof(int), cudaMemcpyDeviceToHost));        //     l ++;
                l ++;
                iterationk ++;
            }
        }else if(pos > 0){
            int done = 1;
            update_visit_by_core0_balance<<<BLK_NUMS, BLK_DIM>>>(core0, p.visit, p.num_vtx, k, p.core); // 这个在while循环外面
            while(done){
                
                iterationh ++;
                cudaMemset(global_done, 0, sizeof(int));  
                cudaMemset(change, 0, sizeof(int) * p.num_vtx); 
                cudaMemset(count_out_s, 0, sizeof(int) * BLK_DIM); 
                cudaMemset(count_out_l, 0, sizeof(int) * BLK_DIM); 
                cudaMemset(count_out_m, 0, sizeof(int) * BLK_DIM); 

                cudaMemset(count_in_s, 0, sizeof(int) * BLK_DIM);
                cudaMemset(count_in_l, 0, sizeof(int) * BLK_DIM);
                cudaMemset(count_in_m, 0, sizeof(int) * BLK_DIM);

                vertex_to_buffer_by_out_degree<<<1024, 256>>>(p.visit, p.num_vtx, k, p.core, p.out_deg, out_buffer_s, out_buffer_m, out_buffer_l, count_out_s, count_out_l, count_out_m);

                // hout_calculate_thread
                // hout_calculate_warp
                // hout_calculate_block


                vertex_to_buffer_by_in_degree<<<1024, 256>>>(p.visit, p.num_vtx, k, p.core, p.out_deg, in_buffer_s, in_buffer_m, in_buffer_l, count_in_s, count_in_l, count_in_m);

            }
        }
 

    }


}