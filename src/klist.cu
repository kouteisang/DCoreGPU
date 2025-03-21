#include "klist.cuh"


__global__ void scan_level(int* t_in_deg, int num_vtx, int* global_buffer, int* buf_count, int level){

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


__global__ void update_level(int* global_buffer, int* buf_count, int* global_count, int* t_in_deg, int* t_out_deg, int* out_offset, int *out_adj, int level){
    
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


   // if (visit[v] == 0 && (t_out_deg[v] == l || t_in_deg[v] < k)) {
        //     int pos = atomicAdd(&sh_buf_count, 1);
        //     t_global_buffer[pos] = v;
        //     visit[v] = 1;
        //     t_out_deg[v] = l;
        // }

__global__ void calculate_scan(int* t_in_deg, int *t_out_deg, int* visit, int num_vtx, int* global_buffer, int* buf_count, int k, int l, int* core){

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


__global__ void calculate_update(int* global_buffer, int* buf_count, int* global_count, 
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
                // if(atomicOr(&visit[o_u], 0) == 0){ // unvisit
                // if(visit[o_u] == 0){
                    int in_deg_u = atomicSub(&t_in_deg[o_u], 1);
                    if(in_deg_u == k && atomicCAS(&visit[o_u], 0, 1) == 0){
                        int end_pos = atomicAdd(&end, 1);
                        t_global_buffer[end_pos] = o_u;
                        core[o_u] = l;
                    }
                // }
            }
            __syncwarp();
            if(i_uid < i_offset_end){
                int i_u = in_adj[i_uid];
                // if(atomicOr(&visit[i_u], 0) == 0 && t_out_deg[i_u] > l){ // unvisit 
                // if(visit[i_u] == 0 && t_out_deg[i_u] > l){ // unvisit  
                    int out_deg_u = atomicSub(&t_out_deg[i_u], 1);
                    if(out_deg_u == (l+1) && atomicCAS(&visit[i_u], 0, 1) == 0){
                        int end_pos = atomicAdd(&end, 1); 
                        t_global_buffer[end_pos] = i_u;
                        core[i_u] = l;
                    }
                //     if(out_deg_u <= l){
                //         atomicAdd(&t_out_deg[i_u], 1);
                //     } 
                // }
            }
        }

        // while(true){
        //     __syncwarp();
        //     if(o_offset_start >= o_offset_end) break;
        //     int o_uid = o_offset_start + lane_id;
        //     o_offset_start = o_offset_start + WARP_SIZE; // update the offset position, each thread maintain its own offset_start
        //     if(o_uid >= o_offset_end) continue;
        //     int o_u = out_adj[o_uid];
        //     if(t_in_deg[o_u] >= k && t_out_deg[o_u] > l){ // unvisit  
        //         int in_deg_u = atomicSub(&t_in_deg[o_u], 1);
        //         if(in_deg_u == k){
        //             int end_pos = atomicAdd(&end, 1);
        //             t_global_buffer[end_pos] = o_u;
        //             visit[o_u] = l; 
        //             // t_out_deg[o_u] = l; 
        //             // atomicExch(&t_out_deg[o_u], l);
        //         }
        //     }
        // }


    //    while(true){
    //         __syncwarp();
    //         if(o2_offset_start >= o2_offset_end) break;
    //         int o2_uid = o2_offset_start + lane_id;
    //         o2_offset_start = o2_offset_start + WARP_SIZE; // update the offset position, each thread maintain its own offset_start
    //         if(o2_uid >= o2_offset_end) continue;
    //         int o2_u = out_adj[o2_uid];
    //         if(visit[o2_u] == l && t_out_deg[o2_u] > l){
    //             t_out_deg[o2_u] = l;
    //         }
    //     }
 
     
        // while (true){
        //     __syncwarp();
        //     if(i_offset_start >= i_offset_end) break;
        //     int i_uid = i_offset_start + lane_id;
        //     i_offset_start = i_offset_start + WARP_SIZE;
        //     if(i_uid >= i_offset_end) continue;
        //     int i_u = in_adj[i_uid];
        //     if(t_in_deg[i_u] >= k && t_out_deg[i_u] > l){ // unvisit
        //         int out_deg_u = atomicSub(&t_out_deg[i_u], 1);
        //         if(out_deg_u == (l+1)){
        //             int end_pos = atomicAdd(&end, 1); 
        //             t_global_buffer[end_pos] = i_u;
        //             visit[i_u] = l;
        //         }
        //         if(out_deg_u <= l){
        //             atomicAdd(&t_out_deg[i_u], 1);
        //             visit[i_u] = l;
        //         }
        //     }
        // }
    }
    
    if(threadIdx.x == 0 && end > 0){
        atomicAdd(global_count, end);
    }

}


__global__ void update_out_deg(int* t_out_deg, int* core, int num_vtx, int* visit, int l){

    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for(int v = tid; v < num_vtx; v += BLK_DIM * BLK_NUMS){
        if(visit[v] == 1 && core[v] == l){
            t_out_deg[v] = l;
        }
    } 
}

void klist_de(G_pointers &p){

    int level = 0;
    int count = 0;
    int* global_count = 0;
    chkerr(cudaMalloc(&global_count, sizeof(int)));

    int* buf_count;
    chkerr(cudaMalloc(&buf_count, sizeof(int) * BLK_NUMS));
    cudaMemset(buf_count, 0, sizeof(int) * BLK_NUMS);

    int* global_buffer;
    chkerr(cudaMalloc(&global_buffer, sizeof(int) * BLK_NUMS * BUFFER_SIZE));
    

    while(count < p.num_vtx){
        cudaMemset(buf_count, 0, sizeof(int) * BLK_NUMS);
        scan_level<<<BLK_NUMS, BLK_DIM>>>(p.t_in_deg, p.num_vtx, global_buffer, buf_count, level);
        update_level<<<BLK_NUMS, BLK_DIM>>>(global_buffer, buf_count, global_count, p.t_in_deg, p.t_out_deg, p.out_offset, p.out_adj, level); 
        chkerr(cudaMemcpy(&count, global_count, sizeof(int), cudaMemcpyDeviceToHost));
        level ++;
    }

    cout << "level = " << level-1 << endl;

    // int *k_res = new int[p.num_vtx];
    // chkerr(cudaMemcpy(k_res, p.t_in_deg, sizeof(int)*(p.num_vtx), cudaMemcpyDeviceToHost)); 
    // for(int i = 0; i < p.num_vtx; i ++){
    //     cout << i << " " << k_res[i] << endl;
    // }

    // std::ifstream file("/home/cheng/DCoreGPU/dataset/papertest/vtx2id.txt");  // 打开文件
    // unordered_map<int, int> id2vtx;
    // int vtx, id;
    // // 逐行读取数据
    // while (file >> vtx >> id) {
    //     id2vtx[id] = vtx;
    // }
    // for(int vv = 0; vv < p.num_vtx; vv ++){
    //     cout << id2vtx[vv] << " " << k_res[vv] << endl;
    // }

 

    // Store the res
    // int** res = new int*[level];
    // for(int l = 0; l < level; l ++){
    //     res[l] = new int[p.num_vtx];
    // }



    int l = 0;
    count = 0;
    for(int k = 0; k < level; k ++){
        cudaMemset(p.core, -1, p.num_vtx * sizeof(int));
        chkerr(cudaMemcpy(p.t_in_deg, p.in_deg, p.num_vtx * sizeof(int), cudaMemcpyDeviceToDevice));
        chkerr(cudaMemcpy(p.t_out_deg, p.out_deg, p.num_vtx * sizeof(int), cudaMemcpyDeviceToDevice));
        cudaMemset(p.visit, 0, p.num_vtx * sizeof(int)); // flag = false means has not visited
        cudaMemset(buf_count, 0, sizeof(int) * BLK_NUMS);
        cudaMemset(global_count, 0, sizeof(int));
        count = 0;
        l = 0;

        while(count < p.num_vtx){
            // cudaMemset(buf_count, 0, sizeof(int) * BLK_NUMS);
            calculate_scan<<<BLK_NUMS, BLK_DIM>>>(p.t_in_deg, p.t_out_deg, p.visit, p.num_vtx, global_buffer, buf_count, k, l, p.core); // scan to find the invalid vertex
            calculate_update<<<BLK_NUMS, BLK_DIM>>>(global_buffer, buf_count, global_count, p.t_in_deg, p.in_adj, p.in_offset, p.t_out_deg, p.out_adj, p.out_offset, p.visit, k, l, p.core);// peel the invalid vertex
            // update_out_deg<<<BLK_NUMS, BLK_DIM>>>(p.t_out_deg, p.core, p.num_vtx, p.visit, l);
            chkerr(cudaMemcpy(&count, global_count, sizeof(int), cudaMemcpyDeviceToHost));        //     l ++;
            l ++;
        }
        // cout << "k = " << k << endl;
        // chkerr(cudaMemcpy(res[k], p.core, p.num_vtx * sizeof(int), cudaMemcpyDeviceToHost));
    }

    

    // // cout << "num of vertes = " << p.num_vtx << endl;
    // // cout << "count = " << count << endl;
    // // cout << "level = " << level << endl;
    // int* in_degree_res = new int[p.num_vtx];
    // chkerr(cudaMemcpy(in_degree_res, p.t_in_deg, p.num_vtx * sizeof(int), cudaMemcpyDeviceToHost));


    // // Save to local
    // std::ifstream file("/home/cheng/DCoreGPU/dataset/my/vtx2id.txt");  // 打开文件
    // unordered_map<int, int> id2vtx;
    // int vtx, id;
    // // 逐行读取数据
    // while (file >> vtx >> id) {
    //     id2vtx[id] = vtx;
    // }

    // for(int k = 0; k < level; k ++){
    //     std::ofstream wr("/home/cheng/DCoreGPU/dataset/my/my-"+std::to_string(k)+"-gpu.txt");

    //     for(int v = 0; v < p.num_vtx; v ++){
    //         wr << id2vtx[v] << " " << res[k][v] << std::endl;
    //     }
    // }
    
}