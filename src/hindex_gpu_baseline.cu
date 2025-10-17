#include "./gpuutilis.cuh"
#include "./hindex_gpu_baseline.cuh"

#define CUDA_CHECK(call) do {                                   \
  cudaError_t err = (call);                                     \
  if (err != cudaSuccess) {                                     \
    fprintf(stderr, "CUDA error %s at %s:%d: %s\n",             \
            #call, __FILE__, __LINE__, cudaGetErrorString(err));\
    exit(1);                                                    \
  }                                                             \
} while (0)

__global__ void vertex_to_buffer_hindex_baseline(int num_vtx, int* global_buffer, int* buf_count, int* visit){

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


__global__ void hindex_out_by_histogram(int* global_buffer, int* buf_count, int* upper, int* core0, int* hindex_out, int* out_adj, int* out_offset, int k, int* his_out, int* doffset){

    __shared__ int end;
    __shared__ int* t_global_buffer;


     if(threadIdx.x == 0){
        t_global_buffer = global_buffer + blockIdx.x * BUFFER_SIZE;
        end = buf_count[blockIdx.x]; // The end position of the buffer
        assert(t_global_buffer!=NULL);
    } 
    __syncthreads();

    for(int vid = threadIdx.x; vid < end; vid += BLK_DIM){
        
        int v = t_global_buffer[vid];
        int offset_start = out_offset[v]; // offset of v
        int offset_end = out_offset[v+1]; // offset of v
        int* histogram = his_out + doffset[v];
        int sum = 0;

        if (upper[v] <= 0) { hindex_out[v] = 0; continue; }

        for (int i = 0; i <= upper[v]; ++i) histogram[i] = 0;

        for(int uid = offset_start; uid < offset_end; uid ++){
            int u = out_adj[uid];
            if(core0[u] >= k && upper[u] >= 0){   
                int bin = upper[u];
                if(bin > upper[v]) bin = upper[v];
                histogram[bin] ++;
            }
        }
        int ans = 0;
        for(int pos = upper[v]; pos >= 0; pos --){
            sum += histogram[pos];
            if(sum >= pos) {
                ans = pos;
                break;
            }
        }
        hindex_out[v] = ans;
    }

}


__global__ void hindex_in_by_histogram(int* global_buffer, int* buf_count, int* upper, int* core0, int* hindex_in, int* in_adj, int* in_offset, int k, int* his_in, int* doffset){

    __shared__ int end;
    __shared__ int* t_global_buffer;


     if(threadIdx.x == 0){
        t_global_buffer = global_buffer + blockIdx.x * BUFFER_SIZE;
        end = buf_count[blockIdx.x]; // The end position of the buffer
        assert(t_global_buffer!=NULL);
    } 
    __syncthreads();

    for(int vid = threadIdx.x; vid < end; vid += BLK_DIM){
        
        int v = t_global_buffer[vid];
        int offset_start = in_offset[v]; // offset of v
        int offset_end = in_offset[v+1]; // offset of v
        int* histogram = his_in + doffset[v];
        int sum = 0;

        if (upper[v] <= 0) { hindex_in[v] = 0; continue; }

        for (int i = 0; i <= upper[v]; ++i) histogram[i] = 0;

        for(int uid = offset_start; uid < offset_end; uid ++){
            int u = in_adj[uid];
            if(core0[u] >= k && upper[u] >= 0){
                int bin = upper[u];
                if(bin > upper[v]) bin = upper[v];
                histogram[bin] ++;
            }
        }

        int ans = 0;
        for(int pos = upper[v]; pos >= 0; pos --){
            sum += histogram[pos];
            if(sum >= k) {
                ans = pos;
                break;
            }
        }
        hindex_in[v] = ans;


    }

}


__global__ void update_upper_by_thread_histogram(int* global_buffer, int* buf_count, int* hindex_in, int* hindex_out, int* core){
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



__global__ void gh_update_change_status_out_histogram(int* global_buffer, int* buf_count, int* hindex_in, int* hindex_out, int* core, int* out_adj, int* out_offset, int* core0, int k, int * global_done, int* visit){
    
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

    for(int vid = threadIdx.x; vid < end; vid += BLK_DIM){
        
        int v = t_global_buffer[vid];
        int minhindex = min(hindex_in[v], hindex_out[v]);
        if(core[v] <= minhindex) continue;
        int offset_start = out_offset[v]; // offset of v
        int offset_end = out_offset[v+1]; // offset of v

        for(int uid = offset_start; uid < offset_end; uid ++){
            int u = out_adj[uid];
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

__global__ void gh_update_change_status_in_histogram(int* global_buffer, int* buf_count, int* hindex_in, int* hindex_out, int* core, int* in_adj, int* in_offset, int* core0, int k, int * global_done, int* visit){
    
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

    for(int vid = threadIdx.x; vid < end; vid += BLK_DIM){
        
        int v = t_global_buffer[vid];
        int minhindex = min(hindex_in[v], hindex_out[v]);
        if(core[v] <= minhindex) continue;
        int offset_start = in_offset[v]; // offset of v
        int offset_end = in_offset[v+1]; // offset of v

        for(int uid = offset_start; uid < offset_end; uid ++){
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


__global__ void vertex_to_buffer_by_core0_thread_histogram(int k, int* core0, int* global_buffer, int* buf_count, int num_vtx){

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


__global__ void hin_count_thread_buffer_ghthread_histogram(int* in_count_num, int* global_buffer, int* buf_count, int* core, int* core0, int* in_adj, int* in_offset, int k){

    __shared__ int end;
    __shared__ int* t_global_buffer;


     if(threadIdx.x == 0){
        t_global_buffer = global_buffer + blockIdx.x * BUFFER_SIZE;
        end = buf_count[blockIdx.x]; // The end position of the buffer
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

void hindex_baseline(G_pointers &p, Graph &g){
    
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

    printf("p.numvtx = %d\n", p.num_vtx);
    cudaMemset(hindex_in, 0, sizeof(int) * p.num_vtx); // 每一个点in的hindex checked
    cudaMemset(hindex_out, 0, sizeof(int) * p.num_vtx); // 每一个点out的hindex  checked 

    int pos = 0;
    int h_kstatus_v_len = h_kstatus_v.size();

    
    int* h_dout = g.get_h_out_deg();

    // new offset in cpu;
    size_t e = 0;
    int* hoffset = new int[p.num_vtx+1];
    hoffset[0] = 0;
    for(int v = 0; v < p.num_vtx; v ++){
        hoffset[v+1] = hoffset[v] + h_dout[v] + 1;
        e += h_dout[v];
    }

    int* doffset;
    chkerr(cudaMalloc(&doffset, sizeof(int) * (p.num_vtx + 1)));

    cudaMemcpy(doffset, hoffset, (p.num_vtx+1) * sizeof(int), cudaMemcpyHostToDevice);

    int* his_out;
    chkerr(cudaMalloc(&his_out, sizeof(int) * (p.num_vtx + e)));

    int* his_in;
    chkerr(cudaMalloc(&his_in, sizeof(int) * (p.num_vtx + e)));

    cudaMemcpy(p.core, p.out_deg, (p.num_vtx) * sizeof(int), cudaMemcpyDeviceToDevice);
    printf("Hello am I here \n");

     while(pos < h_kstatus_v_len){
        // printf("pos = %d\n", pos);
        int h_min = INT_MAX; 
        int k = h_kstatus_v[pos];
        cudaMemset(p.visit, 0, p.num_vtx * sizeof(int)); // flag = false means has not visited
        cudaMemset(p.in_count_num, -1, p.num_vtx * sizeof(int));

        if(pos >= 0){
            
            int done = 1;

            update_visit_by_core0_balance_buffer_utils<<<BLK_NUMS, BLK_DIM>>>(core0, p.visit, p.num_vtx, k, p.core, p.in_count_num); // 这个在while循环外面
            
            while(done){
                
                if(pos > 0) iterationh ++;
                cudaMemset(buf_count, 0, sizeof(int) * BLK_NUMS); // buf count  checked
                cudaMemset(global_done, 0, sizeof(int));  // 是否完成  checked

                // cudaMemset(his_out, 0, sizeof(int) * (p.num_vtx + e));
                // cudaMemset(his_in, 0, sizeof(int) * (p.num_vtx + e));
                

                vertex_to_buffer_hindex_baseline<<<BLK_NUMS, BLK_DIM>>>(p.num_vtx, global_buffer, buf_count, p.visit); // 将需要改变的放在buffer里面并设置visit = 1 checked
                hindex_out_by_histogram<<<BLK_NUMS, BLK_DIM>>>(global_buffer, buf_count, p.core, core0, hindex_out, p.out_adj, p.out_offset, k, his_out, doffset);      
                hindex_in_by_histogram<<<BLK_NUMS, BLK_DIM>>>(global_buffer, buf_count, p.core, core0, hindex_in, p.in_adj, p.in_offset, k, his_in, doffset);
           

                chkerr(cudaMemset(p.visit, 0, sizeof(int) * p.num_vtx));

                gh_update_change_status_out_histogram<<<BLK_NUMS, BLK_DIM>>>(global_buffer, buf_count, hindex_in, hindex_out, p.core, p.out_adj, p.out_offset, core0, k, global_done, p.visit);

                gh_update_change_status_in_histogram<<<BLK_NUMS, BLK_DIM>>>(global_buffer, buf_count, hindex_in, hindex_out, p.core, p.in_adj, p.in_offset, core0, k, global_done, p.visit);

                update_upper_by_thread_histogram<<<BLK_NUMS, BLK_DIM>>>(global_buffer, buf_count, hindex_in, hindex_out, p.core);


                chkerr(cudaMemcpy(&done, global_done, sizeof(int), cudaMemcpyDeviceToHost));

            }
        }
        
        if(pos + 1 < h_kstatus_v_len && h_kstatus_v[pos+1] != k+1){
            
            cudaMemcpy(d_min, &max_val, sizeof(int), cudaMemcpyHostToDevice);

            cudaMemset(buf_count, 0, sizeof(int) * BLK_NUMS); // buf count  checked

            vertex_to_buffer_by_core0_thread_histogram<<<BLK_NUMS, BLK_DIM>>>(k, core0, global_buffer, buf_count, p.num_vtx);

            hin_count_thread_buffer_ghthread_histogram<<<BLK_NUMS, BLK_DIM>>>(p.in_count_num, global_buffer, buf_count, p.core, core0, p.in_adj, p.in_offset, k);
            
            reduceMinkernel_balance_buffer_utils<<< (p.num_vtx+256-1)/256, 256>>>(p.in_count_num, d_min, p.num_vtx);

            cudaMemcpy(&h_min, d_min, sizeof(int), cudaMemcpyDeviceToHost);

            if(h_min != INT_MAX && pos+1 < h_kstatus_v_len && h_min+1 < h_kstatus_v[pos+1]){
                h_kstatus_v.insert(h_kstatus_v.begin() + pos + 1, h_min+1);
                h_kstatus_v_len ++;
                // cout << h_min+1 << " is inserted into the list " << endl;
            }
        }
        pos ++;
    }

    std::cout << "h_kstatus_v_len = " << h_kstatus_v.size() << std::endl;
    cout << "iterationh = " << iterationh << endl;
    cout << "iterationk = " << iterationk << endl;
    cout << "total iteration = " << iterationk + iterationh << endl;

    

}