#include "./klist_balance_buffer_atomic.cuh"
#include "./gpuutilis.cuh"



__global__ void hout_calculate_thread_buffer_atomic(int* out_buffer_s, int* count_out_s, int* upper, int* core0, int* hindex_out, int* out_adj, int* out_offset, int k){

    __shared__ int end;
    __shared__ int* t_global_buffer;


     if(threadIdx.x == 0){
        t_global_buffer = out_buffer_s + blockIdx.x * BUFFER_SIZE;
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


__global__ void hout_calculate_warp_buffer_atomic(int* out_buffer_m, int* count_out_m, int* upper, int* core0, int* hindex_out, int* out_adj, int* out_offset, int k){
  
    __shared__ int start, end;
    __shared__ int* t_global_buffer;
    __shared__ int best_mid[BLK_DIM/32];
    __shared__ bool flag[BLK_DIM/32];
    __shared__ int warp_total[BLK_DIM/32];
    

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
            warp_total[warp_id] = 0;
        }

        __syncwarp();
        
            
        while(true){
            if(lane_id == 0){
                warp_total[warp_id] = 0;
            }
            __syncwarp();
            if(flag[warp_id] == false || best_mid[warp_id] == 0) break;
            __syncwarp();
            int mid = best_mid[warp_id];

            for(int uid = offset_start+lane_id; uid < offset_end; uid += WARP_SIZE){
                int u = out_adj[uid];
                if (core0[u] >= k && upper[u] >= mid){
                    atomicAdd(&warp_total[warp_id],1);
                }
            }

            __syncwarp();

            if(lane_id == 0){
                if(warp_total[warp_id] >= mid){
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


__global__ void hout_calculate_block_buffer_atomic(int* out_buffer_l, int* count_out_l, int* upper, int* core0, int* hindex_out, int* out_adj, int* out_offset, int k, int* block_count_out){
  
    __shared__ int res_shared;
    // __shared__ int block_count;


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
        
        if(threadIdx.x == 0){
            // block_count = 0;
            block_count_out[vid%BLK_NUMS] = 0;
        }
        __syncthreads();

        while(res >= 0){
            
            if(threadIdx.x == 0){
                // block_count = 0;
                block_count_out[vid%BLK_NUMS] = 0;
            }
            __syncthreads();
            for(int uid = offset_start + threadIdx.x; uid < offset_end; uid += blockDim.x){
                int u = out_adj[uid];
                if(core0[u] >= k && upper[u] >= res){
                    atomicAdd(&block_count_out[vid%BLK_NUMS], 1);
                }
            }
            __syncthreads(); 

            if(tid == 0){
                if (block_count_out[vid%BLK_NUMS] >= res) {
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


__global__ void hin_calculate_thread_buffer_atomic(int* in_buffer_s, int* count_in_s, int* upper, int* core0, int* hindex_in, int* in_adj, int* in_offset, int k, int* hindex_out){

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
        int offset_start = in_offset[v]; // offset of v
        int offset_end = in_offset[v+1]; // offset of v

        int local_count = 0;
        int res = upper[v];
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


__global__ void hin_calculate_warp_buffer_atomic(int* in_buffer_m, int* count_in_m, int* upper, int* core0, int* hindex_in, int* in_adj, int* in_offset, int k, int* hindex_out){
   
    __shared__ int start, end;
    __shared__ int* t_global_buffer;
    __shared__ int best_mid[BLK_DIM/32];
    __shared__ bool flag[BLK_DIM/32];
    __shared__ int warp_total[BLK_DIM/32];
    

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
            best_mid[warp_id] = upper[v];
            flag[warp_id] = true;
        }

        __syncwarp();
        
            
        while(true){
            if(lane_id == 0) {
                warp_total[warp_id] = 0;
            }
            __syncwarp();
            if(flag[warp_id] == false || best_mid[warp_id] == 0) break;
            __syncwarp();
            int mid = best_mid[warp_id];
        
            for(int uid = offset_start+lane_id; uid < offset_end; uid += WARP_SIZE){
                int u = in_adj[uid];
                if(core0[u] >= k && upper[u] >= mid){
                    atomicAdd(&warp_total[warp_id], 1);
                }
            }
            __syncwarp();

            if(lane_id == 0){
                if(warp_total[warp_id] >= k){
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


__global__ void hin_calculate_block_buffer_atomic(int* in_buffer_l, int* count_in_l, int* upper, int* core0, int* hindex_in, int* in_adj, int* in_offset, int k, int* hindex_out, int* block_count_in){
  
    __shared__ int res_shared;
    // __shared__ int block_count;

    int warp_per_block = blockDim.x / WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int tid = threadIdx.x;

    int total = count_in_l[0];


    for(int vid = blockIdx.x; vid < total; vid += gridDim.x){

        if(threadIdx.x == 0){
            // block_count = 0;
            block_count_in[vid % BLK_NUMS] = 0;
        }
        __syncthreads();

        int v = in_buffer_l[vid]; // Get the vertex id

        int offset_start = in_offset[v]; // offset of v
        int offset_end = in_offset[v+1]; // offset of v

        int res = upper[v];

        while(res >= 0){
            if(threadIdx.x == 0){
                // block_count = 0;
                block_count_in[vid % BLK_NUMS] = 0;
            }
            __syncthreads();
            for(int uid = offset_start + threadIdx.x; uid < offset_end; uid += blockDim.x){
                int u = in_adj[uid];
                if(core0[u] >= k && upper[u] >= res){
                    atomicAdd(&block_count_in[vid % BLK_NUMS], 1);
                }
            }

            __syncthreads(); 

            if(tid == 0){
                if (block_count_in[vid % BLK_NUMS] >= k) {
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

void klist_balance_buffer_atomic_de(G_pointers &p){

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

    bool* kstatus;
    chkerr(cudaMalloc(&kstatus, kmax * sizeof(bool)));
    bool* h_kstatus = new bool[kmax];
    kstatus_update_utils<<<BLK_NUMS, BLK_DIM>>>(p.t_in_deg, kstatus, p.num_vtx);
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
    
    int* block_count_out;
    cudaMalloc(&block_count_out, sizeof(int)* BLK_NUMS);

    int* block_count_in;
    cudaMalloc(&block_count_in, sizeof(int)* BLK_NUMS);


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
                scan_phase_balance_buffer_utils<<<BLK_NUMS, BLK_DIM>>>(p.t_in_deg, p.t_out_deg, p.visit, p.num_vtx, global_buffer, buf_count, k, l, p.core); // scan to find the invalid vertex
                update_phase_balance_buffer_utils<<<BLK_NUMS, BLK_DIM>>>(global_buffer, buf_count, global_count, p.t_in_deg, p.in_adj, p.in_offset, p.t_out_deg, p.out_adj, p.out_offset, p.visit, k, l, p.core);// peel the invalid vertex
                chkerr(cudaMemcpy(&count, global_count, sizeof(int), cudaMemcpyDeviceToHost));        //     l ++;
                l ++;
                iterationk ++;
            }
        }else if(pos > 0){
            int done = 1;
            update_visit_by_core0_balance_buffer_utils<<<BLK_NUMS, BLK_DIM>>>(core0, p.visit, p.num_vtx, k, p.core, p.in_count_num); // 这个在while循环外面
            cudaMemset(buf_count, 0, sizeof(int) * BLK_NUMS); // buf count  checked
            vertex_to_buffer_buffer_utils<<<BLK_NUMS, BLK_DIM>>>(p.num_vtx, global_buffer, buf_count, p.visit);

            while(done){

                iterationh ++;
                // printf("iteration h = %d\n", iterationh);
                cudaMemset(global_done, 0, sizeof(int));  

                cudaMemset(count_out_s, 0, sizeof(int) * BLK_NUMS); 
                cudaMemset(count_out_m, 0, sizeof(int) * BLK_NUMS); 
                cudaMemset(count_out_l, 0, sizeof(int));
                cudaMemset(block_count_out, 0, sizeof(int) * BLK_NUMS);

                cudaMemset(count_in_s, 0, sizeof(int) * BLK_NUMS); 
                cudaMemset(count_in_m, 0, sizeof(int) * BLK_NUMS); 
                cudaMemset(count_in_l, 0, sizeof(int));
                cudaMemset(block_count_in, 0, sizeof(int) * BLK_NUMS);

                vertex_to_buffer_by_out_degree_buffer_utils<<<BLK_NUMS, BLK_DIM>>>(p.visit, p.num_vtx, k, p.core, p.out_deg, out_buffer_s, out_buffer_m, out_buffer_l, count_out_s, count_out_m, count_out_l, global_buffer, buf_count);
                vertex_to_buffer_by_in_degree_buffer_utils<<<BLK_NUMS, BLK_DIM>>>(p.visit, p.num_vtx, k, p.core, p.in_deg, in_buffer_s, in_buffer_m, in_buffer_l, count_in_s, count_in_m, count_in_l, global_buffer, buf_count);


                hout_calculate_thread_buffer_atomic<<<BLK_NUMS, BLK_DIM>>>(out_buffer_s, count_out_s, p.core, core0, hindex_out, p.out_adj, p.out_offset, k);
                hout_calculate_warp_buffer_atomic<<<BLK_NUMS, BLK_DIM>>>(out_buffer_m, count_out_m, p.core, core0, hindex_out, p.out_adj, p.out_offset, k);
                hout_calculate_block_buffer_atomic<<<BLK_NUMS, BLK_DIM>>>(out_buffer_l, count_out_l, p.core, core0, hindex_out, p.out_adj, p.out_offset, k, block_count_out);

                hin_calculate_thread_buffer_atomic<<<BLK_NUMS, BLK_DIM>>>(in_buffer_s, count_in_s, p.core, core0, hindex_in, p.in_adj, p.in_offset, k, hindex_out);
                hin_calculate_warp_buffer_atomic<<<BLK_NUMS, BLK_DIM>>>(in_buffer_m, count_in_m, p.core, core0, hindex_in, p.in_adj, p.in_offset, k, hindex_out);
                hin_calculate_block_buffer_atomic<<<BLK_NUMS, BLK_DIM>>>(in_buffer_l, count_in_l, p.core, core0, hindex_in, p.in_adj, p.in_offset, k, hindex_out, block_count_in);

                cudaMemset(p.visit, 0, sizeof(int) * p.num_vtx); 
                

                update_change_status_out_thread_utils<<<BLK_NUMS, BLK_DIM>>>(out_buffer_s, count_out_s, hindex_in, hindex_out, p.core, p.out_adj, p.out_offset, core0, k, global_done, p.visit);
                update_change_status_out_warp_utils<<<BLK_NUMS, BLK_DIM>>>(out_buffer_m, count_out_m, hindex_in, hindex_out, p.core, p.out_adj, p.out_offset, core0, k, global_done, p.visit);
                update_change_status_out_block_utils<<<BLK_NUMS, BLK_DIM>>>(out_buffer_l, count_out_l, hindex_in, hindex_out, p.core, p.out_adj, p.out_offset, core0, k, global_done, p.visit);

                update_change_status_in_thread_utils<<<BLK_NUMS, BLK_DIM>>>(in_buffer_s, count_in_s, hindex_in, hindex_out, p.core, p.in_adj, p.in_offset, core0, k, global_done, p.visit);
                update_change_status_in_warp_utils<<<BLK_NUMS, BLK_DIM>>>(in_buffer_m, count_in_m, hindex_in, hindex_out, p.core, p.in_adj, p.in_offset, core0, k, global_done, p.visit);
                update_change_status_in_block_utils<<<BLK_NUMS, BLK_DIM>>>(in_buffer_l, count_in_l, hindex_in, hindex_out, p.core, p.in_adj, p.in_offset, core0, k, global_done, p.visit);
                
                update_upper_by_out_buffer_s_utils<<<BLK_NUMS, BLK_DIM>>>(out_buffer_s, count_out_s, hindex_in, hindex_out, p.core);
                update_upper_by_out_buffer_m_utils<<<BLK_NUMS, BLK_DIM>>>(out_buffer_m, count_out_m, hindex_in, hindex_out, p.core);
                update_upper_by_out_buffer_l_utils<<<BLK_NUMS, BLK_DIM>>>(out_buffer_l, count_out_l, hindex_in, hindex_out, p.core);
          
                cudaMemcpy(&done, global_done, sizeof(int), cudaMemcpyDeviceToHost);
            }
        }

        if(pos + 1 < h_kstatus_v_len && h_kstatus_v[pos+1] != k+1){
            cudaMemcpy(d_min, &max_val, sizeof(int), cudaMemcpyHostToDevice);

            cudaMemset(count_in_s, 0, sizeof(int) * BLK_NUMS); 
            cudaMemset(count_in_m, 0, sizeof(int) * BLK_NUMS); 
            cudaMemset(count_in_l, 0, sizeof(int));

            vertex_to_buffer_by_core0_buffer_utils<<<BLK_NUMS, BLK_DIM>>>(k, core0, p.in_deg, in_buffer_s, in_buffer_m, in_buffer_l, count_in_s, count_in_m, count_in_l, global_buffer, buf_count);
            hin_count_thread_buffer_utils<<<BLK_NUMS, BLK_DIM>>>(p.in_count_num, in_buffer_s, count_in_s, p.core, core0, p.in_adj, p.in_offset, k);
            hin_count_warp_buffer_utils<<<BLK_NUMS, BLK_DIM>>>(p.in_count_num, in_buffer_m, count_in_m, p.core, core0, p.in_adj, p.in_offset, k);
            hin_count_block_buffer_utils<<<BLK_NUMS, BLK_DIM>>>(p.in_count_num, in_buffer_l, count_in_l, p.core, core0, p.in_adj, p.in_offset, k);
            
            reduceMinkernel_balance_buffer_utils<<< (p.num_vtx+256-1)/256, 256>>>(p.in_count_num, d_min, p.num_vtx);

            cudaMemcpy(&h_min, d_min, sizeof(int), cudaMemcpyDeviceToHost);
            // cout << "k = " << k << ", h_min = " << h_min << endl;

            if(h_min != INT_MAX && pos+1 < h_kstatus_v_len && h_min+1 < h_kstatus_v[pos+1]){
                h_kstatus_v.insert(h_kstatus_v.begin() + pos + 1, h_min+1);
                h_kstatus_v_len ++;
                // cout << h_min+1 << " is inserted into the list " << endl;
            }
        }
        pos ++;

    }

    cout << "iterationh = " << iterationh << endl;
    cout << "iterationk = " << iterationk << endl;
    cout << "total iteration = " << iterationk + iterationh << endl;

}