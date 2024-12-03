#include "klist.cuh"


__global__ void scan_level(int* t_in_deg, int num_vtx, int* global_buffer, int* buf_count, int level){

    // printf("%d\n", p.num_vtx);
    __shared__ int* t_global_buffer;
    __shared__ int sh_buf_count;
    if(threadIdx.x == 0){
        sh_buf_count = 0;
        t_global_buffer = global_buffer + blockIdx.x * BUFFER_SIZE;
    }
    __syncthreads();

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for(int v = tid; v < num_vtx; v += blockDim.x*gridDim.x){
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
    __shared__ int start_prime, end_prime;
    __shared__ int* t_global_buffer;

    int warp_per_block = blockDim.x / WARP_SIZE;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int v;
    if(threadIdx.x == 0){
        t_global_buffer = global_buffer + blockIdx.x * BUFFER_SIZE;
        start = 0; // The begin position of the global_buffer
        end = buf_count[blockIdx.x]; // The end position of the buffer
        start_prime = start;
        end_prime = end;
        printf("start = %d, end = %d, start_prime =  %d, end_prime = %d, %d, %d\n", start, end, start_prime, end_prime, t_global_buffer[0], t_global_buffer[1]);
    } 

    __syncthreads();

    while(true){
        __syncthreads();
        if(start == end) break; // All the thread break the iteration
        start_prime = start + warp_id; // Get the vertex id position
        end_prime = end; // Get the last position of the vertex id
        __syncthreads();
        if(start_prime >= end_prime) continue; // The vertex id is larger than the number of valid vertices in the buffer
        if(threadIdx.x == 0){
            start = min(start + blockDim.x / WARP_SIZE, end); // update the start position
        }
        __syncthreads();
        int v = t_global_buffer[start_prime]; // Get the vertex id
        // if(tid == 0 || tid == 32) 
        // printf("start_prime = %d, v = %d\n", start_prime, v);
        int offset_start = out_offset[v]; // offset of v 
        int offset_end = out_offset[v+1]; // offset of v
        
        while (true){
            __syncwarp();
            if(offset_start >= offset_end) break;
            int uid = offset_start + lane_id;
            offset_start = offset_start + 32; // update the offset position
            if(uid >= offset_end) continue; // This vertex does not has so many neighbouthood
            int u = out_adj[uid]; // v's out-neighbouthood u
            if(t_in_deg[u] > level){
                int in_deg_u = atomicSub(&t_in_deg[u], 1);
                int end_pos = atomicAdd(&end, 1);
                if(in_deg_u == level+1) t_global_buffer[end_pos] = u;
                if(in_deg_u <= level) { // Add it back
                    atomicAdd(&t_in_deg[u], 1);
                }
            }

        }   
    }

    // if(threadIdx.x == 0){
    //     printf("end = %d", end);
    //     atomicAdd(global_count, end);
    // }


}


void klist_de(G_pointers &p){

    int level = 1;
    int count = 0;
    int* global_count = 0;
    chkerr(cudaMalloc(&global_count, sizeof(int)));

    int* buf_count;
    chkerr(cudaMalloc(&buf_count, sizeof(int) * BLK_NUMS));
    cudaMemset(buf_count, 0, sizeof(int) * BLK_NUMS);

    int* global_buffer;
    chkerr(cudaMalloc(&global_buffer, sizeof(int) * BLK_NUMS * BUFFER_SIZE));


   int* t_buf_count;
    t_buf_count = new int[BLK_NUMS];

    int* t_buffer = new int[BUFFER_SIZE * BLK_NUMS];
    

    // while(count < p.num_vtx){
        cudaMemset(buf_count, 0, sizeof(int) * BLK_NUMS);
        scan_level<<<BLK_NUMS, BLK_DIM>>>(p.t_in_deg, p.num_vtx, global_buffer, buf_count, level);

        cudaMemcpy(t_buffer, global_buffer, sizeof(int)* BUFFER_SIZE * BLK_NUMS, cudaMemcpyDeviceToHost);
        for(int i = 0; i <= 10; i ++){
            cout << t_buffer[i] << endl;
        }

        update_level<<<BLK_NUMS, BLK_DIM>>>(global_buffer, buf_count, global_count, p.t_in_deg, p.t_out_deg, p.out_offset, p.out_adj, level); 
        cudaMemcpy(&count, global_count, sizeof(int), cudaMemcpyDeviceToHost);
        level ++;
        cout << "Count = " << count << endl;
    // }



     // cudaMemcpy(t_buf_count, buf_count, sizeof(int) * BLK_NUMS, cudaMemcpyDeviceToHost);
        // for(int i = 0; i < BLK_NUMS; i ++){
        //     cout << "BLK_NUMS = " << i << " val = " << t_buf_count[i] << endl;  
        // }
        // cudaDeviceSynchronize();
}