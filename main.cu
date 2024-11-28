#include "header.h"
#include "./src/klist.cuh"
#include "utils.h"


int main(int argc, char* argv[]){

    string dataset = "em";
    string file_path = "/home/cheng/DCoreGPU/dataset/"+ dataset + "/";

    Graph g = Graph(file_path, dataset);

    int num_vtx = g.get_num_vtx();
    ll num_edge = g.get_num_edge();

    G_pointers data_pointers;
    malloc_graph_gpu_memory(g, data_pointers);

    // int* t_in_deg;
    // int* t_out_deg;
    // chkerr(cudaMalloc(&t_in_deg, num_vtx * sizeof(int)));
    // chkerr(cudaMalloc(&t_out_deg, num_vtx * sizeof(int))); 

    // bool* flag;
    // cudaMalloc(&flag, num_vtx * sizeof(bool));

    gpu_data_init(data_pointers);

    klist_de(data_pointers);
    // int count = 0;
    // int level = *std::min_element(g.get_h_in_adj(), g.get_h_in_deg() + num_vtx);
    // int level = 0;
    // while(count < num_vtx){
    //     // scan<<<,bs>>>(in_adj, t_in_deg, in_offset, out_adj, t_out_deg, out_offset, num_vtx);
    // }

    
    
    
    
    return 0;
}