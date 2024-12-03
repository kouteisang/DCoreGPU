#include "utils.h"
#include "./src/klist.cuh"



int main(int argc, char* argv[]){

    string dataset = "example";
    string file_path = "/home/cheng/DCoreGPU/dataset/"+ dataset + "/";

    Graph g = Graph(file_path, dataset);

    int num_vtx = g.get_num_vtx();
    ll num_edge = g.get_num_edge();

    G_pointers data_pointers;
    malloc_graph_gpu_memory(g, data_pointers);

    gpu_data_init(data_pointers);

    klist_de(data_pointers);
    
    
    return 0;
}