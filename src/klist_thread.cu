#include "klist_thread.cuh"
#include "gpuutilis.cuh"


__global__ void update_thread(int* global_buffer, int* buf_count, int* global_count, 
                                int* t_in_deg, int* in_adj, int* in_offset, 
                                int* t_out_deg, int* out_adj, int* out_offset, 
                                int* visit, int k, int l, int* core){
        
    __shared__ int start, end;
    __shared__ int* t_global_buffer;

    // int warp_per_block = blockDim.x / WARP_SIZE;
    // int warp_id = threadIdx.x / WARP_SIZE;
    // int lane_id = threadIdx.x % WARP_SIZE;
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
        start_prime = start + threadIdx.x; // Get the vertex id position
        end_prime = end; // Get the last position of the vertex id
        __syncthreads();
        if(start_prime >= end_prime) continue; // The vertex position is larger than the number of valid vertices in the buffer
        if(threadIdx.x == 0){
            start = min(start + blockDim.x, end); // update the start position
        }
        int v = t_global_buffer[start_prime]; // Get the vertex id

        int o_offset_start = out_offset[v]; // offset of v
        int o_offset_end = out_offset[v+1]; // offset of v
        // int o2_offset_start = o_offset_start; // offset of v
        // int o2_offset_end = o_offset_end; // offset of v
         

        int i_offset_start = in_offset[v];
        int i_offset_end = in_offset[v+1];

        for(int o_uid = o_offset_start; o_uid < o_offset_end; o_uid ++){
            int o_u = out_adj[o_uid];
            int in_deg_u = atomicSub(&t_in_deg[o_u], 1);
            if(in_deg_u == k && atomicCAS(&visit[o_u], 0, 1) == 0){
                int end_pos = atomicAdd(&end, 1);
                t_global_buffer[end_pos] = o_u;
                core[o_u] = l;
            }

        }

        for(int i_uid = i_offset_start; i_uid < i_offset_end; i_uid ++){
            int i_u = in_adj[i_uid];
            int out_deg_u = atomicSub(&t_out_deg[i_u], 1);
            if(out_deg_u == (l+1) && atomicCAS(&visit[i_u], 0, 1) == 0){
                int end_pos = atomicAdd(&end, 1); 
                t_global_buffer[end_pos] = i_u;
                core[i_u] = l;
            }

        }
    }
    
    if(threadIdx.x == 0 && end > 0){
        atomicAdd(global_count, end);
    }

}


void klist_thread(G_pointers &p){


    int iteration = 0;
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
        klistprune_scan_utils<<<BLK_NUMS, BLK_DIM>>>(p.t_in_deg, p.num_vtx, global_buffer, buf_count, level);
        klistprune_update_utils<<<BLK_NUMS, BLK_DIM>>>(global_buffer, buf_count, global_count, p.t_in_deg, p.t_out_deg, p.out_offset, p.out_adj, level); 
        chkerr(cudaMemcpy(&count, global_count, sizeof(int), cudaMemcpyDeviceToHost));
        level ++;
    }
    chkerr(cudaMemcpy(core0, p.t_in_deg, sizeof(int)*p.num_vtx, cudaMemcpyDeviceToDevice));   


    bool* kstatus;
    chkerr(cudaMalloc(&kstatus, level * sizeof(bool)));
    bool* h_kstatus = new bool[level];
    kstatus_update_utils<<<BLK_NUMS, BLK_DIM>>>(p.t_in_deg, kstatus, p.num_vtx);
    chkerr(cudaMemcpy(h_kstatus, kstatus, sizeof(bool)*level, cudaMemcpyDeviceToHost));    

    vector<int> h_kstatus_v;
    for(int i = 0; i < level; i ++){
        if(h_kstatus[i]){
            h_kstatus_v.push_back(i);
        }
    }

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

        while(count < p.num_vtx){
            scan_block_utils<<<BLK_NUMS, BLK_DIM>>>(p.t_in_deg, p.t_out_deg, p.visit, p.num_vtx, global_buffer, buf_count, k, l, p.core); // scan to find the invalid vertex
            update_thread<<<BLK_NUMS, BLK_DIM>>>(global_buffer, buf_count, global_count, p.t_in_deg, p.in_adj, p.in_offset, p.t_out_deg, p.out_adj, p.out_offset, p.visit, k, l, p.core);// peel the invalid vertex
            chkerr(cudaMemcpy(&count, global_count, sizeof(int), cudaMemcpyDeviceToHost));        //     l ++;
            l ++;
            iteration++;
            // std::cout << k << " " << l << std::endl;
        }
        
        if(pos + 1 < h_kstatus_v_len && h_kstatus_v[pos+1] != k+1){
            cudaMemcpy(d_min, &max_val, sizeof(int), cudaMemcpyHostToDevice);
            check_innb_count_utils<<<BLK_NUMS, BLK_DIM>>>(p.in_count_num, p.core, core0, p.num_vtx, p.in_offset, p.in_adj, k);
            reduceMinkernel_utils<<< (p.num_vtx+256-1)/256, 256>>>(p.in_count_num, d_min, p.num_vtx);
            cudaMemcpy(&h_min, d_min, sizeof(int), cudaMemcpyDeviceToHost);

            if(h_min != INT_MAX && pos+1 < h_kstatus_v_len && h_min+1 < h_kstatus_v[pos+1]){
                h_kstatus_v.insert(h_kstatus_v.begin() + pos + 1, h_min+1);
                h_kstatus_v_len ++;
            }
        }
        pos ++;
    
    }

    std::cout << "h_kstatus_v_len = " << h_kstatus_v.size() << std::endl;
    cout << "iteration = " << iteration << endl;
    
}