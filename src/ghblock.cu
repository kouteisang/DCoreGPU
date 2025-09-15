#include "ghblock.cuh"
#include "gpuutilis.cuh"


__device__ int warp_reduce_sum_ghblock(int val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void vertex_to_buffer_gblock(int num_vtx, int* global_buffer, int* buf_count, int* visit){

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


__global__ void hindex_out_cal_block(int* global_buffer, int* buf_count, int* upper, int* core0, int* hindex_out, int* out_adj, int* out_offset, int k){
  
    __shared__ int warp_counts[BLK_DIM/WARP_SIZE]; 
    __shared__ int final_count; 
    __shared__ int res_shared;


    int warp_per_block = blockDim.x / WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int tid = threadIdx.x;


    __shared__ int end;
    __shared__ int* t_global_buffer;

    if(threadIdx.x == 0){
        end = buf_count[blockIdx.x];
        t_global_buffer = global_buffer + blockIdx.x * BUFFER_SIZE;
        assert(t_global_buffer!=NULL);
    }
    __syncthreads();



    for(int vid = 0; vid < end; vid ++){
        int v = t_global_buffer[vid]; // Get the vertex id
        int offset_start = out_offset[v]; // offset of v
        int offset_end = out_offset[v+1]; // offset of v

        int res = (upper[v]);
        

        while(res >= 0){
            int local_count = 0;
            for(int uid = offset_start + threadIdx.x; uid < offset_end; uid += blockDim.x){
                int u = out_adj[uid];
                local_count += (core0[u] >= k && upper[u] >= res);
            }

            int warp_sum = warp_reduce_sum_ghblock(local_count);
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


__global__ void hindex_in_cal_block(int* global_buffer, int* buf_count, int* upper, int* core0, int* hindex_in, int* in_adj, int* in_offset, int k){
  
    __shared__ int warp_counts[BLK_DIM/WARP_SIZE]; 
    __shared__ int final_count; 
    __shared__ int res_shared;


    int warp_per_block = blockDim.x / WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int tid = threadIdx.x;


    __shared__ int end;
    __shared__ int* t_global_buffer;

    if(threadIdx.x == 0){
        end = buf_count[blockIdx.x];
        t_global_buffer = global_buffer + blockIdx.x * BUFFER_SIZE;
        assert(t_global_buffer!=NULL);
    }
    __syncthreads();

    for(int vid = 0; vid < end; vid ++){
        int v = t_global_buffer[vid]; // Get the vertex id
        int offset_start = in_offset[v]; // offset of v
        int offset_end = in_offset[v+1]; // offset of v

        int res = (upper[v]);

        while(res >= 0){
            int local_count = 0;
            for(int uid = offset_start + threadIdx.x; uid < offset_end; uid += blockDim.x){
                int u = in_adj[uid];
                local_count += (core0[u] >= k && upper[u] >= res);
            }

            int warp_sum = warp_reduce_sum_ghblock(local_count);
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

__global__ void ghblock_update_change_status_out(int* global_buffer, int* buf_count, int* hindex_in, int* hindex_out, int* core, int* out_adj, int* out_offset, int* core0, int k, int * global_done, int* visit){
    
    __shared__ int end;
    __shared__ int* t_global_buffer;
    __shared__ int block_has_change;
    
     if(threadIdx.x == 0){
        t_global_buffer = global_buffer + blockIdx.x * BUFFER_SIZE;
        end = buf_count[blockIdx.x]; // The end position of the buffer
        block_has_change = 0;
        assert(t_global_buffer!=NULL);
    } 
    __syncthreads();

    for(int vid = 0; vid < end; vid ++){
        
        int v = t_global_buffer[vid];
        int minhindex = min(hindex_in[v], hindex_out[v]);
        if(core[v] <= minhindex) continue;
        int offset_start = out_offset[v]; // offset of v
        int offset_end = out_offset[v+1]; // offset of v

        for(int uid = offset_start+threadIdx.x; uid < offset_end; uid += blockDim.x){
            int u = out_adj[uid];
            int minu = min(hindex_in[u], hindex_out[u]);
            if (core0[u] >= k && minu > minhindex) {
                visit[u] = 1;
                block_has_change = 1; //to do
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0 && block_has_change) {
        *global_done = 1;
    }
}

__global__ void ghblock_update_change_status_in(int* global_buffer, int* buf_count, int* hindex_in, int* hindex_out, int* core, int* in_adj, int* in_offset, int* core0, int k, int * global_done, int* visit){
    
    __shared__ int end;
    __shared__ int* t_global_buffer;
    __shared__ int block_has_change;
    
     if(threadIdx.x == 0){
        t_global_buffer = global_buffer + blockIdx.x * BUFFER_SIZE;
        end = buf_count[blockIdx.x]; // The end position of the buffer
        block_has_change = 0;
        assert(t_global_buffer!=NULL);
    } 
    __syncthreads();

    for(int vid = 0; vid < end; vid ++){
        
        int v = t_global_buffer[vid];
        int minhindex = min(hindex_in[v], hindex_out[v]);
        if(core[v] <= minhindex) continue;
        int offset_start = in_offset[v]; // offset of v
        int offset_end = in_offset[v+1]; // offset of v

        for(int uid = offset_start+threadIdx.x; uid < offset_end; uid += blockDim.x){
            int u = in_adj[uid];
            int minu = min(hindex_in[u], hindex_out[u]);
            // if(minu > core[v]) continue; // I modified on August 27
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

__global__ void update_upper_by_thread_in_ghblock(int* global_buffer, int* buf_count, int* hindex_in, int* hindex_out, int* core){
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

__global__ void vertex_to_buffer_by_core0_thread_in_ghblock(int k, int* core0, int* global_buffer, int* buf_count, int num_vtx){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int sh_buf_count;
    __shared__ int* t_global_buffer;

    if(threadIdx.x == 0){
        sh_buf_count = 0;
        t_global_buffer = global_buffer + blockIdx.x * BUFFER_SIZE;
    }
    __syncthreads();
    
   for(int v = tid; v < num_vtx; v += BLK_DIM * BLK_NUMS){
        if(core0[v] >= k){
            int pos = atomicAdd(&sh_buf_count, 1);
            t_global_buffer[pos] = v;
        }
    }
    
    __syncthreads();

    if(threadIdx.x == 0){
        buf_count[blockIdx.x] = sh_buf_count;
    }

}


__global__ void hin_count_thread_buffer_ghblock(int* in_count_num, int* global_buffer, int* buf_count, int* core, int* core0, int* in_adj, int* in_offset, int k){

    __shared__ int warp_counts[BLK_DIM/WARP_SIZE]; 
    __shared__ int end;
    __shared__ int* t_global_buffer;

    __shared__ int final_count;

    int warp_per_block = blockDim.x / WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;


     if(threadIdx.x == 0){
        t_global_buffer = global_buffer + blockIdx.x * BUFFER_SIZE;
        end = buf_count[blockIdx.x]; // The end position of the buffer
        assert(t_global_buffer!=NULL);
    } 
    __syncthreads();

    for(int vid = 0; vid < end; vid ++){
        
        int v = t_global_buffer[vid];
        int core_v = core[v];
        int offset_start = in_offset[v]; // offset of v
        int offset_end = in_offset[v+1]; // offset of v

        int local_count = 0;
        int flag = true;

        for(int uid = offset_start+threadIdx.x; uid < offset_end; uid += blockDim.x){
            int u = in_adj[uid];
            local_count += (core0[u] >= k && core[u] >= core_v);
        }
        
        int warp_sum = warp_reduce_sum_ghblock(local_count);
        if(lane_id == 0) warp_counts[warp_id] = warp_sum;
        __syncthreads();

        if(threadIdx.x == 0){
            final_count = 0;
            for (int i = 0; i < warp_per_block; i++) {
                final_count += warp_counts[i];
            }
            in_count_num[v] = final_count;
        }
    }

    

}


void ghblock_decomposition(G_pointers &p){
    
    printf("Hello all, this is the ghblock part\n");
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
        klistprune_scan_utils<<<BLK_NUMS, BLK_DIM>>>(p.t_in_deg, p.num_vtx, global_buffer, buf_count, kmax);
        klistprune_update_utils<<<BLK_NUMS, BLK_DIM>>>(global_buffer, buf_count, global_count, p.t_in_deg, p.t_out_deg, p.out_offset, p.out_adj, kmax); 
        chkerr(cudaMemcpy(&count, global_count, sizeof(int), cudaMemcpyDeviceToHost));
        kmax ++;
    }

    int* core0;
    chkerr(cudaMalloc(&core0, p.num_vtx*sizeof(int)));
    chkerr(cudaMemcpy(core0, p.t_in_deg, sizeof(int)*p.num_vtx, cudaMemcpyDeviceToDevice));     

    cout << "kmax = " << kmax-1 << endl;

    bool* kstatus;
    chkerr(cudaMalloc(&kstatus, kmax * sizeof(bool)));
    bool* h_kstatus = new bool[kmax];
    kstatus_update_utils<<<BLK_NUMS, BLK_DIM>>>(p.t_in_deg, kstatus, p.num_vtx);
    chkerr(cudaMemcpy(h_kstatus, kstatus, sizeof(bool)*kmax, cudaMemcpyDeviceToHost));     

    // int process = 0;
    vector<int> h_kstatus_v;
    for(int i = 0; i < kmax; i ++){
        if(h_kstatus[i]){
            h_kstatus_v.push_back(i);
            // cout << "i = " << i << endl;
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

    int* global_done;
    chkerr(cudaMalloc(&global_done, sizeof(int)));  

    cudaMemset(hindex_in, 0, sizeof(int) * p.num_vtx); // 每一个点in的hindex checked
    cudaMemset(hindex_out, 0, sizeof(int) * p.num_vtx); // 每一个点out的hindex  checked 

    int pos = 0;
    int h_kstatus_v_len = h_kstatus_v.size();


    while(pos < h_kstatus_v_len){
        
        int h_min = INT_MAX; 
        int k = h_kstatus_v[pos];
        cudaMemset(buf_count, 0, sizeof(int) * BLK_NUMS);
        cudaMemset(p.visit, 0, p.num_vtx * sizeof(int)); // flag = false means has not visited
        cudaMemset(p.in_count_num, -1, p.num_vtx * sizeof(int));

        if(pos == 0){
            count = 0;
            l = 0;
            while(count < p.num_vtx){
                scan_phase_balance_buffer_utils<<<BLK_NUMS, BLK_DIM>>>(p.t_in_deg, p.t_out_deg, p.visit, p.num_vtx, global_buffer, buf_count, k, l, p.core); // scan to find the invalid vertex
                update_phase_balance_buffer_utils<<<BLK_NUMS, BLK_DIM>>>(global_buffer, buf_count, global_count, p.t_in_deg, p.in_adj, p.in_offset, p.t_out_deg, p.out_adj, p.out_offset, p.visit, k, l, p.core);// peel the invalid vertex
                chkerr(cudaMemcpy(&count, global_count, sizeof(int), cudaMemcpyDeviceToHost));        //     l ++;
                l ++;
                iterationk ++;
            }
        }else if(pos > 0){
            int done = 1;
            // printf("pos = %d\n", pos);
            update_visit_by_core0_balance_buffer_utils<<<BLK_NUMS, BLK_DIM>>>(core0, p.visit, p.num_vtx, k, p.core, p.in_count_num); // 这个在while循环外面
            while(done){
                iterationh ++;
                cudaMemset(buf_count, 0, sizeof(int) * BLK_NUMS); // buf count  checked
                cudaMemset(global_done, 0, sizeof(int));  // 是否完成  checked

                vertex_to_buffer_gblock<<<BLK_NUMS, BLK_DIM>>>(p.num_vtx, global_buffer, buf_count, p.visit); // 将需要改变的放在buffer里面并设置visit = 1 checked

                hindex_out_cal_block<<<BLK_NUMS, BLK_DIM>>>(global_buffer, buf_count, p.core, core0, hindex_out, p.out_adj, p.out_offset, k);
                hindex_in_cal_block<<<BLK_NUMS, BLK_DIM>>>(global_buffer, buf_count, p.core, core0, hindex_in, p.in_adj, p.in_offset, k);

                chkerr(cudaMemset(p.visit, 0, sizeof(int) * p.num_vtx));

                ghblock_update_change_status_out<<<BLK_NUMS, BLK_DIM>>>(global_buffer, buf_count, hindex_in, hindex_out, p.core, p.out_adj, p.out_offset, core0, k, global_done, p.visit);
                ghblock_update_change_status_in<<<BLK_NUMS, BLK_DIM>>>(global_buffer, buf_count, hindex_in, hindex_out, p.core, p.in_adj, p.in_offset, core0, k, global_done, p.visit);

                update_upper_by_thread_in_ghblock<<<BLK_NUMS, BLK_DIM>>>(global_buffer, buf_count, hindex_in, hindex_out, p.core);
                chkerr(cudaMemcpy(&done, global_done, sizeof(int), cudaMemcpyDeviceToHost));
            }
        }

        if(pos + 1 < h_kstatus_v_len && h_kstatus_v[pos+1] != k+1){
            
            cudaMemcpy(d_min, &max_val, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemset(buf_count, 0, sizeof(int) * BLK_NUMS); // buf count  checked
            vertex_to_buffer_by_core0_thread_in_ghblock<<<BLK_NUMS, BLK_DIM>>>(k, core0, global_buffer, buf_count, p.num_vtx);

            hin_count_thread_buffer_ghblock<<<BLK_NUMS, BLK_DIM>>>(p.in_count_num, global_buffer, buf_count, p.core, core0, p.in_adj, p.in_offset, k);
            
            reduceMinkernel_balance_buffer_utils<<< (p.num_vtx+256-1)/256, 256>>>(p.in_count_num, d_min, p.num_vtx);
            cudaMemcpy(&h_min, d_min, sizeof(int), cudaMemcpyDeviceToHost);

            if(h_min != INT_MAX && pos+1 < h_kstatus_v_len && h_min+1 < h_kstatus_v[pos+1]){
                h_kstatus_v.insert(h_kstatus_v.begin() + pos + 1, h_min+1);
                h_kstatus_v_len ++;
            }

        }
        pos ++;
    }

    std::cout << "h_kstatus_v_len = " << h_kstatus_v.size() << std::endl;
    cout << "iterationh = " << iterationh << endl;
    cout << "iterationk = " << iterationk << endl;
    cout << "total iteration = " << iterationk + iterationh << endl;
}